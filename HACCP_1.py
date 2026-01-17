# ============================
# HCCP – Validades (Streamlit)
# ============================

# --- IMPORTS ---
import re
from datetime import datetime, timedelta, time
import os
import io
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo
from PIL import Image
import numpy as np
import cv2
import pytesseract
from rapidfuzz import process, fuzz
import socket


pytesseract.pytesseract.tesseract_cmd = "/opt/local/bin/tesseract"


TZ = ZoneInfo("Europe/Lisbon")

ZEBRAS = {
    "ZQ1": {
        "nome": "Zebra ZQ610 159",
        "ip": "10.151.60.159",
        "dpi": 203,
    },
    "ZQ2": {
        "nome": "Zebra ZQ610 158",
        "ip": "10.151.60.158",
        "dpi": 203,
    },
    "ZQ3": {
        "nome": "Zebra ZQ610 157",
        "ip": "10.151.60.157",
        "dpi": 203,
    },

}


def normalizar_nome_produto(txt: str) -> str:
    if not txt:
        return ""

    s = txt.lower().strip()

    # remove coisas entre parênteses (ex: (CDP), (200-600G), etc.)
    s = re.sub(r"\([^)]*\)", " ", s)

    # remove pesos e unidades (ex: 120g, 1.280kg, 200-600g, 1280 g)
    s = re.sub(r"\b\d+[.,]?\d*\s*(kg|g|gr|gramas)\b", " ", s)
    s = re.sub(r"\b\d+\s*-\s*\d+\s*(kg|g|gr)\b", " ", s)

    # remove números soltos
    s = re.sub(r"\b\d+\b", " ", s)

    # remove pontuação extra
    s = re.sub(r"[^a-zà-ÿ\s]", " ", s)

    # normaliza espaços
    s = re.sub(r"\s+", " ", s).strip()

    return s


def preprocess_gray(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Aumenta tamanho para OCR (muito importante)
    h, w = gray.shape[:2]
    scale = 2 if max(h, w) < 1400 else 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reduz ruído leve
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def preprocess_thresh(pil_img: Image.Image) -> np.ndarray:
    gray = preprocess_gray(pil_img)

    # Threshold adaptativo (melhor em sombras)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    # Limpeza leve
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th


def ocr_best_of_two(pil_img: Image.Image) -> str:
    # OCR em dois modos e escolhe o melhor
    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

    t1 = pytesseract.image_to_string(preprocess_gray(pil_img), lang="por", config=config) or ""
    t2 = pytesseract.image_to_string(preprocess_thresh(pil_img), lang="por", config=config) or ""

    return t1 if len(t1) >= len(t2) else t2


def auto_crop_document(pil_img: Image.Image) -> Image.Image:
    """
    Tenta detectar o maior retângulo (papel) e recortar.
    Se falhar, devolve a imagem original.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # suaviza e detecta bordas
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # fecha pequenos buracos nas bordas
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # encontra contornos
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img

    # pega o maior contorno
    cnt = max(cnts, key=cv2.contourArea)

    # recorte por bounding box (simples e robusto)
    x, y, w, h = cv2.boundingRect(cnt)

    # evita recortes muito pequenos (falha)
    H, W = gray.shape[:2]
    if w < 0.3 * W or h < 0.3 * H:
        return pil_img

    cropped = img[y:y+h, x:x+w]
    return Image.fromarray(cropped)


def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # AUMENTA tamanho (OCR melhora muito)
    h, w = gray.shape[:2]
    scale = 2 if max(h, w) < 1200 else 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Aumenta contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Remove ruído leve
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarização adaptativa (melhor em papel com sombras)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    # Pequena "limpeza" morfológica
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return th


def ocr_image_to_text(pil_img: Image.Image) -> str:
    processed = preprocess_for_ocr(pil_img)

    # 6 = bloco de texto; 4 também é bom para "colunas"
    config = "--oem 1 --psm 6"

    text = pytesseract.image_to_string(processed, lang="por", config=config)
    return text or ""


def extrair_nome_do_padroes(linhas: list[str]) -> str | None:
    """
    Extrai o nome após 'Denominação Comercial' (mesma linha ou em duas linhas),
    até antes de 'Nome Científico' (ou outras seções).
    """
    if not linhas:
        return None

    low = [l.lower().strip() for l in linhas]

    start_idx = None

    # Caso A: "denominação comercial" na mesma linha
    for i, l in enumerate(low):
        if "denomina" in l and "comercial" in l:
            start_idx = i + 1
            break

    # Caso B: "denominação" numa linha e "comercial" na seguinte
    if start_idx is None:
        for i in range(len(low) - 1):
            if "denomina" in low[i] and "comercial" in low[i + 1]:
                start_idx = i + 2
                break

    if start_idx is None:
        return None

    nome_partes = []
    for j in range(start_idx, len(linhas)):
        lj = low[j]
        if not linhas[j].strip():
            break
        # para quando chegar em outras secções
        if ("nome cient" in lj or "método" in lj or "metodo" in lj or
            "país" in lj or "pais" in lj or "lote" in lj):
            break

        nome_partes.append(linhas[j].strip())

    nome = " ".join(nome_partes).strip()
    return nome if nome else None


def extrair_lote(linhas: list[str]) -> str | None:
    for l in linhas:
        m = re.search(r"\blote\b\s*[:\-]?\s*([A-Z0-9]+)", l, flags=re.I)
        if m:
            return m.group(1).strip()
    return None


def extrair_validade_ultimos4(linhas: list[str]) -> str | None:
    """
    Regra que você passou:
    - validade está na ÚLTIMA linha
    - são os últimos 4 números
    - 0000 => avaliação macroscópica
    - ex: 1804 => 18/04
    """
    # pega última linha "com conteúdo"
    ultima = None
    for l in reversed(linhas):
        if l.strip():
            ultima = l.strip()
            break
    if not ultima:
        return None

    # captura últimos 4 dígitos do fim (ignora pontos e letras)
    m = re.search(r"(\d{4})\D*$", ultima)
    if not m:
        # fallback: procura qualquer grupo de 4 dígitos no fim do texto
        joined = " ".join(linhas)
        m = re.search(r"(\d{4})\D*$", joined)
        if not m:
            return None

    ult4 = m.group(1)

    if ult4 == "0000":
        return "avaliação macroscópica"

    # ddmm -> dd/mm
    dd = ult4[:2]
    mm = ult4[2:]
    return f"{dd}/{mm}"


def sugerir_produto_catalogo(nome_extraido: str, produtos: list["Produto"]) -> str | None:
    if not nome_extraido:
        return None

    nome_norm = normalizar_nome_produto(nome_extraido)

    catalogo = [p.nome for p in produtos if p.nome]
    if not catalogo:
        return None

    # também normaliza o catálogo para comparar "igual com igual"
    catalogo_norm = {p.nome: normalizar_nome_produto(p.nome) for p in produtos if p.nome}

    # fuzzy match em cima das versões normalizadas
    escolhas = list(catalogo_norm.values())
    melhor_norm, score, idx = process.extractOne(
        nome_norm,
        escolhas,
        scorer=fuzz.WRatio
    )

    # recupera o nome ORIGINAL do catálogo correspondente ao "melhor_norm"
    nome_original = None
    for original, norm in catalogo_norm.items():
        if norm == melhor_norm:
            nome_original = original
            break

    return nome_original if score >= 80 else None


def parse_data_forn(txt: str) -> datetime | None:
    if not txt:
        return None

    txt = txt.strip().lower()

    # caso especial
    if "avalia" in txt and "macro" in txt:
        return None

    fmts = ["%d/%m/%Y %H:%M", "%d/%m/%Y", "%d/%m/%y", "%d/%m"]  # <- adicionado "%d/%m"

    for f in fmts:
        try:
            dt = datetime.strptime(txt, f)

            # Se vier sem ano (%d/%m), usa o ano atual (heurística)
            if f == "%d/%m":
                now = datetime.now(TZ)
                dt = dt.replace(year=now.year)

                # se a data já "passou" muito, assume próximo ano
                # (ajuda em virada de ano / etiquetas do início do ano)
                candidate = dt.replace(tzinfo=TZ)
                if candidate < now - timedelta(days=7):
                    dt = dt.replace(year=now.year + 1)

            if f in ("%d/%m/%Y", "%d/%m/%y", "%d/%m"):
                dt = datetime.combine(dt.date(), time(23, 59, 59))

            return dt.replace(tzinfo=TZ)

        except ValueError:
            continue

    return None

#impressão com a Zebra


def cm_to_dots(cm: float, dpi: int) -> int:
    inches = cm / 2.54
    return int(round(inches * dpi))

def build_haccp_zpl(executar: "Produto", dpi: int = 203) -> str:
    pw = 640   # 8 cm
    ll = 305   # 3.8 cm

    lines = [
        "VALIDADE HACCP",
        f"PRODUTO: {executar.nome}",
        f"LOTE: {executar.lote}",
        f"VAL FORN: {executar.val_for}",
        f"EXP: {executar.agora.strftime('%d/%m %H:%M')}",
        f"VAL: {executar.validade.strftime('%d/%m %H:%M')}",
    ]

    if executar.obs:
        lines.append(f"OBS: {executar.obs}")

    text = "\\&".join(lines)  # quebra de linha ZPL

    zpl = f"""
^XA
^CI28
^PW{pw}
^LL{ll}
^LH0,0

^FO15,15
^A0N,22,22
^FB{pw-30},7,3,L,0
^FD{text}^FS

^XZ
""".strip()

    return zpl


def send_zpl_to_zebra(ip: str, zpl: str, port: int = 9100, timeout: float = 3.0) -> None:
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.sendall(zpl.encode("utf-8"))
    except socket.timeout:
        raise ConnectionError("A impressora não respondeu (timeout).")
    except ConnectionRefusedError:
        raise ConnectionError("Conexão recusada pela impressora.")
    except OSError as e:
        raise ConnectionError(f"Impressora não encontrada na rede ({ip}).")




# --------------------------------------------------------------------
# 1) CONFIGURAÇÃO INICIAL DO STREAMLIT
#    -> Precisa ser a PRIMEIRA chamada do Streamlit no script
# --------------------------------------------------------------------
st.set_page_config(page_title="HCCP - Validades", page_icon="🧾", layout="centered")

## ----------------------
# Tela inicial
# ----------------------
if "iniciado" not in st.session_state:
    st.session_state.iniciado = False

if not st.session_state.iniciado:
    st.title("Validades HACCP dos Frescos ")

    st.markdown(
        """
        Bem-vindo ao sistema de **Validades HACCP**!  

        Sou **Rafael Cavalcante Salvador**, colaborador da **secção dos frescos no Continente Bom Dia Santo Amaro (Oeiras)**
        e atualmente estudante de um curso profissional de Programação em Python.  

        Durante a rotina de trabalho, percebi que a elaboração das validades HACCP, feita manualmente, podia gerar erros e retrabalho.
        Pensando nisso, desenvolvi este programa com o objetivo de **tornar o processo mais ágil, fiável e acessível** para toda a equipa.

        A aplicação permite calcular e gerar validades HACCP de forma simples e intuitiva, reduzindo falhas humanas e garantindo que
        mesmo colaboradores com menos experiência na secção consigam realizar o procedimento de maneira correta.

        Este projeto une a prática do dia a dia no setor de frescos com os conhecimentos que venho adquirindo em programação,
        e espero que seja útil no apoio à rotina e no reforço da qualidade e segurança alimentar.
        """
    )

    if st.button("👉 Iniciar aplicação", type="primary"):
        st.session_state.iniciado = True
        st.rerun()

    # 🔒 Impede que o resto do script rode antes de clicar em "Iniciar"
    st.stop()

# --------------------------------------------------------------------
# 2) LOCALIZAÇÃO DO ARQUIVO EXCEL
#    -> Usamos caminho absoluto com base na pasta deste .py
#       Assim, mesmo que o diretório de trabalho mude, o arquivo é achado.
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "produtos_hccp_master.xlsx")


# --------------------------------------------------------------------
# 3) CLASSE DE DOMÍNIO: Produto
#    -> Representa um item do Excel e também serve como “rascunho”
#       enquanto o usuário gera uma validade (objeto 'executar').
# --------------------------------------------------------------------
class Produto:
    def __init__(self, nome=None, armazenamento=None, exposto=None, obs=None):
        # Dados “estáticos” do produto (vindos do Excel):
        self.nome = (nome or "").strip()
        self.exposto = exposto                       # regra de exposição (ex.: "24 horas", "produto do dia")
        self.armazenamento = armazenamento           # regra de armazenamento (pode ser None)
        self.obs = obs                               # observações

        # Dados “dinâmicos” usados no fluxo HCCP:
        self.lote = ""                               # lote digitado pelo usuário
        self.val_for = ""                            # validade do fornecedor (digitada pelo usuário)
        self.agora = None                            # data/hora de geração do documento
        self.validade = None                         # data/hora de validade calculada

    # Constrói um texto simples para exibir dados do produto (consulta)
    def texto_vizualizar(self) -> str:
        return f"""
------------------
     PRODUTO
NOME: {self.nome}
VALIDADE EXPOSTO: {self.exposto}
VALIDADE ARMAZENAMENTO: {self.armazenamento}
OBS: {self.obs}
------------------"""

    # Desenha o bloco de “consulta” no Streamlit + botão para baixar TXT
    @staticmethod
    def _bloco_vizualizar(obj: "Produto"):
        bloco = obj.texto_vizualizar()
        # st.text mostra texto monoespaçado sem borda (poderia usar st.code também)
        st.code(bloco, language ="text")
        # Download do bloco como .txt
        st.download_button(
            "Baixar como TXT",
            data=bloco.encode("utf-8"),
            file_name=f"Produto_{obj.nome[:20].replace(' ', '_')}.txt",
            mime="text/plain",
        )



    # Tela de “Consultar produto” (busca por selectbox)
        # Tela de “Consultar produto” (busca por selectbox)
    @staticmethod
    def pesquisart(lista: list["Produto"]):
        st.subheader("Pesquisar produto")

        # Monta as opções do select.
        nomes = [""] + [obj.nome for obj in lista]

        # selectbox retorna a string escolhida
        escolha = st.selectbox(
            "Escolha o produto",
            options=nomes,
            index=0,
            key="pesq_nome_sel",
        )

        col1, col2 = st.columns([1, 1])

        # 1. AÇÃO PESQUISAR: Salva o produto encontrado na session_state
        with col1:
            if st.button("Pesquisar", type="primary"):
                st.session_state["produto_pesquisado"] = None  # Zera primeiro
                if escolha != "":
                    for obj in lista:
                        if obj.nome == escolha:
                            # Salva o produto na sessão para persistir a visualização
                            st.session_state["produto_pesquisado"] = obj
                            break
                else:
                    st.warning("Selecione um produto na lista.")

        # 2. AÇÃO NOVA PESQUISA: Limpa a sessão
        with col2:
            def _nova_pesquisa():
                st.session_state["pesq_nome_sel"] = ""  # limpa o selectbox
                st.session_state["produto_pesquisado"] = None  # limpa o produto salvo
                # Se usou st.session_state para a mensagem de impressão, limpe-a também.
                # st.session_state.pop("imprimir_msg", None) # Não é mais necessário com st.toast

            st.button("Nova pesquisa", on_click=_nova_pesquisa)

        # 3. EXIBIÇÃO PERSISTENTE: Mostra o produto se ele estiver salvo na sessão
        produto_a_exibir = st.session_state.get("produto_pesquisado")
        if produto_a_exibir:
            Produto._bloco_vizualizar(produto_a_exibir)
    # Constrói o texto final do HCCP (com validade e campos do rascunho 'executar')
    def texto_vizualizar_hccp(self) -> str:
        # Usamos “or ''” para evitar aparecer “None” caso algum campo não esteja preenchido
        return f"""
------------------
VALIDADE HACCP:
PRODUTO: {self.nome or ''}
LOTE: {self.lote or ''}
VALIDADE FORNECEDOR: {self.val_for or ''}
EXPOSTO: {self.agora.strftime('Data: %d/%m/%y|| Hora: %H:%M') if self.agora else ''}
VALIDADE: {self.validade.strftime('Data: %d/%m/%y|| Hora: %H:%M') if self.validade else ''}
OBS: {self.obs or ''}
------------------"""

    # Tela “Gerar validade HACCP”
    @staticmethod
    def hccp_streamlit(executar: "Produto", lista: list["Produto"]):
        st.subheader("Gerar validade HACCP")

        # 1) Select de produtos (nomes vindos da lista já filtrada por seção)
        nomes = [obj.nome for obj in lista]
        artigo = st.selectbox(
            "Digite o nome do produto",
            options=nomes,
            index=0 if nomes else None,
            key="hccp_artigo",
        )

        # 2) Inputs do usuário (LOTE e VALIDADE DO FORNECEDOR)
        #    -> Guardamos direto nos atributos do objeto 'executar'
        executar.lote = st.text_input("Digite o lote:", key="hccp_lote")
        executar.val_for = st.text_input("Digite a validade do fornecedor:", key="hccp_val_for")

        # 3) Momento atual (usado tanto para exibir quanto para calcular validade)
        executar.agora = datetime.now(TZ)

        # 4) Ao clicar em “Gerar Validade” calculamos a validade e montamos o bloco
        if st.button("Gerar Validade", type="primary"):
            # IMPORTANTE: zere SOMENTE o que vai recalcular (a validade).
            # NÃO apague lote/val_for/nome/obs aqui, pois são entradas do usuário ou dados do produto.
            executar.validade = None

            # Localiza o produto selecionado e copia nome/obs do “catálogo” para o rascunho 'executar'
            for obj in lista:
                if artigo.lower() == obj.nome.lower():
                    executar.nome = obj.nome
                    executar.obs = obj.obs
                    texto = (obj.exposto or "").lower()

                    # Interpreta a regra: “X dias” ou “Y horas” ou “produto do dia”
                    dias = re.search(r"(\d+)\s*dias?", texto)
                    horas = re.search(r"(\d+)\s*horas?", texto)

                    if "produto do dia" in texto:
                        # Até 23:59:59 do dia atual
                        executar.validade = datetime(
                            executar.agora.year, executar.agora.month, executar.agora.day, 23, 59, 59, tzinfo=TZ
                        )
                    elif dias:
                        executar.validade = executar.agora + timedelta(days=int(dias.group(1)))
                    elif horas:
                        executar.validade = executar.agora + timedelta(hours=int(horas.group(1)))
                    # Se não cair em nenhum caso, “validade” permanece None
                    break

            # --- aplicar limite do fornecedor / ou usar só fornecedor se não houver "normal" ---
            dt_forn = parse_data_forn(executar.val_for)

            if dt_forn and executar.validade:
                # fique com a data mais cedo (mais restritiva)
                if dt_forn < executar.validade:
                    executar.validade = dt_forn
                    executar.obs = (executar.obs or "")
                    if "Fornecedor limitou validade" not in executar.obs:
                        executar.obs = (executar.obs + " | Fornecedor limitou validade").strip(" |")
            elif dt_forn and not executar.validade:
                # não houve regra "normal" -> use a data do fornecedor como validade final
                executar.validade = dt_forn

            dt_forn = parse_data_forn(executar.val_for)
            if dt_forn and executar.val_for:
                if dt_forn < executar.validade:
                    executar.validade = dt_forn


            # 5) Exibição do bloco + opção de download
        if getattr(executar, "validade", None) is not None and (executar.lote or "").strip():
            bloco = executar.texto_vizualizar_hccp()
            st.success("Validade gerada com sucesso!")
            st.code(bloco, language="text")

            # -------------------------
            # IMPRESSÃO ZEBRA
            # -------------------------
            imp_id = st.selectbox(
                "Impressora Zebra",
                options=list(ZEBRAS.keys()),
                format_func=lambda k: f"{k} - {ZEBRAS[k]['nome']}",
            )

            if st.button("Imprimir validade", key="btn_print_tab2"):
                zebra = ZEBRAS[imp_id]
                zpl = build_haccp_zpl(executar_tab3, dpi=zebra["dpi"])
                try:
                    send_zpl_to_zebra(zebra["ip"], zpl)
                    st.success("Etiqueta enviada para a impressora com sucesso ✅")

                except ConnectionError as e:
                    st.warning(
                        f"⚠️ Não foi possível imprimir.\n\n"
                        f"Impressora: {zebra['nome']}\n"
                        f"Motivo: {str(e)}\n\n"
                        "Verifique se a impressora está ligada e conectada à rede."
                    )

                except Exception as e:
                    st.error("❌ Erro inesperado ao tentar imprimir.")
                    st.exception(e)

            # Baixar como TXT (nome do arquivo inclui produto + lote + val_for)
            st.download_button(
                "Baixar como TXT",
                data=bloco.encode("utf-8"),
                file_name=f"HCCP_{executar.nome[:20].replace(' ', '_')}_{executar.lote}_{executar.val_for}.txt",
                mime="text/plain",
            )



            if not (executar.lote or "").strip():
                st.warning("Informe o lote.")


        # 6) Botão “Gerar nova validade”
        #    -> Limpa os inputs do usuário e esconde o bloco (zerando a validade),
        #       mas NÃO apaga dados de catálogo nem mexe no produto escolhido.
        def _gerar_nova_validade():
            st.session_state["hccp_lote"] = ""      # limpa campo de texto do lote
            st.session_state["hccp_val_for"] = ""   # limpa campo da validade do fornecedor
            executar.validade = None                # esconde bloco gerado
            # opcional: para resetar também o select do produto, descomente:
            # st.session_state.pop("hccp_artigo", None)

        st.button("Gerar nova validade", on_click=_gerar_nova_validade)


# --------------------------------------------------------------------
# 4) LEITURA DO EXCEL
#    -> Carrega a planilha única (colunas esperadas: Secao, Nome, Armazenamento, Exposto, Obs).
#    -> Retorna o DataFrame completo e a lista de seções.
#    -> cache_data evita reler o arquivo a cada interação.
# --------------------------------------------------------------------
@st.cache_data
def ler_excel(path=EXCEL_PATH):
    if not os.path.exists(path):
        st.error(f"Arquivo Excel não encontrado: {path}")
        st.stop()

    try:
        df = pd.read_excel(path, engine="openpyxl").fillna("")
    except Exception as e:
        st.error("Falha ao ler o Excel.")
        st.exception(e)
        st.stop()

    # Valida colunas obrigatórias
    obrig = {"Secao", "Nome", "Armazenamento", "Exposto", "Obs"}
    faltando = obrig - set(df.columns)
    if faltando:
        st.error(f"Colunas faltando no Excel: {sorted(faltando)}")
        st.stop()

    secoes = sorted(df["Secao"].dropna().unique().tolist())
    return df, secoes


# Carrega o banco (uma vez, por cache)
df_all, secoes = ler_excel()



# 5) FILTRA A SEÇÃO E CONVERTE PARA LISTA DE Produto
def para_lista_de_produtos(df: pd.DataFrame, secao: str | None = None) -> list[Produto]:
    if secao:
        df = df[df["Secao"].str.lower() == secao.lower()]
    return [
        Produto(
            nome=row["Nome"],
            armazenamento=(row["Armazenamento"] or ''),
            exposto=row["Exposto"],
            obs=row["Obs"],
        )
        for _, row in df.iterrows()
    ]


# Seção escolhida pelo usuário
secao_escolhida = st.sidebar.selectbox("Selecione a secção", secoes)

# Lista de produtos apenas da seção selecionada
produtos = para_lista_de_produtos(df_all, secao_escolhida)


# --------------------------------------------------------------------
# 6) CICLO DE VIDA DO OBJETO 'executar'
#    -> Criamos (uma vez) e reusamos entre interações via session_state
# --------------------------------------------------------------------
if "executar_tab2" not in st.session_state:
    st.session_state.executar_tab2 = Produto()

if "executar_tab3" not in st.session_state:
    st.session_state.executar_tab3 = Produto()

executar_tab2 = st.session_state.executar_tab2
executar_tab3 = st.session_state.executar_tab3


# --------------------------------------------------------------------
# 7) UI PRINCIPAL: título + abas
# --------------------------------------------------------------------
st.code(
    """
-------------------------
VALIDADES HACCP DOS FRESCOS
-------------------------
Desenvolvido por Rafael Salvador
""".strip(),
    language="text",
)

tab1, tab2, tab3 = st.tabs(["Consultar validade", "Gerar validade HACCP", "Gerar por foto"])


with tab1:
    Produto.pesquisart(produtos)

with tab2:
    Produto.hccp_streamlit(executar_tab2, produtos)


with tab3:
    st.subheader("Gerar validade por foto (rastreabilidade)")

    # -------------------------
    # ESTADO PERSISTENTE
    # -------------------------
    if "foto_bytes" not in st.session_state:
        st.session_state.foto_bytes = None

    if "foto_haccp_pronto" not in st.session_state:
        st.session_state.foto_haccp_pronto = False

    if "print_feedback" not in st.session_state:
        # {'kind': 'success'|'warning'|'error', 'msg': '...'}
        st.session_state.print_feedback = None

    # -------------------------
    # FEEDBACK DO PRINT (sempre visível no tab3)
    # -------------------------
    fb = st.session_state.print_feedback
    if fb:
        if fb["kind"] == "success":
            st.success(fb["msg"])
        elif fb["kind"] == "warning":
            st.warning(fb["msg"])
        else:
            st.error(fb["msg"])

    # -------------------------
    # ENTRADA DA IMAGEM
    # -------------------------
    col1, col2 = st.columns(2)
    with col1:
        img_camera = st.camera_input("Tirar foto", key="foto_camera")
    with col2:
        img_upload = st.file_uploader("Carregar foto (PNG/JPG)", type=["png", "jpg", "jpeg"], key="foto_upload")

    img_file = img_upload or img_camera
    if img_file is not None:
        st.session_state.foto_bytes = img_file.getvalue()

    pil_img = None
    if st.session_state.foto_bytes:
        pil_img = Image.open(io.BytesIO(st.session_state.foto_bytes)).convert("RGB")

    # -------------------------
    # OCR + EXTRAÇÃO (só se houver imagem)
    # -------------------------
    if pil_img:
        st.image(pil_img, caption="Imagem recebida", use_container_width=True)

        usar_auto = st.toggle("Recortar automaticamente a rastreabilidade", value=True, key="toggle_auto_tab3")
        pil_ocr = auto_crop_document(pil_img) if usar_auto else pil_img
        st.image(pil_ocr, caption="Imagem usada no OCR", use_container_width=True)

        with st.spinner("A ler o documento (OCR)..."):
            texto = ocr_best_of_two(pil_ocr)

        st.markdown("### Texto")
        st.code(texto, language="text")

        linhas = [l.strip() for l in texto.splitlines() if l.strip()]

        nome_extraido = extrair_nome_do_padroes(linhas)
        lote = extrair_lote(linhas)
        val_for = extrair_validade_ultimos4(linhas)

        nome_sugerido = sugerir_produto_catalogo(nome_extraido or "", produtos)

        st.markdown("### Campos detectados")
        colA, colB = st.columns(2)
        st.write("**Nome normalizado (para match):**", normalizar_nome_produto(nome_extraido or "") or "—")

        with colA:
            st.write("**Nome (OCR):**", nome_extraido or "—")
            st.write("**Nome sugerido (catálogo):**", nome_sugerido or "—")
        with colB:
            st.write("**Lote:**", lote or "—")
            st.write("**Validade fornecedor:**", val_for or "—")

        st.markdown("---")
        st.markdown("### Confirmar / ajustar")

        nomes_catalogo = [p.nome for p in produtos]
        nome_final = st.selectbox(
            "Produto do catálogo",
            options=nomes_catalogo,
            index=nomes_catalogo.index(nome_sugerido) if nome_sugerido in nomes_catalogo else 0,
            key="foto_produto_final",
        )

        lote_final = st.text_input("Lote", value=lote or "", key="foto_lote_final")
        val_for_final = st.text_input("Validade do fornecedor", value=val_for or "", key="foto_valfor_final")

        # -------------------------
        # GERAR (ação)
        # -------------------------
        if st.button("Gerar validade (a partir da foto)", type="primary", key="btn_gerar_tab3"):
            st.session_state.print_feedback = None  # limpa feedback anterior

            for obj in produtos:
                if obj.nome == nome_final:
                    executar_tab3.nome = obj.nome
                    executar_tab3.obs = obj.obs
                    executar_tab3.exposto = obj.exposto
                    break

            executar_tab3.lote = lote_final
            executar_tab3.val_for = val_for_final
            executar_tab3.agora = datetime.now(TZ)
            executar_tab3.validade = None

            regra = (executar_tab3.exposto or "").lower()
            dias = re.search(r"(\d+)\s*dias?", regra)
            horas = re.search(r"(\d+)\s*horas?", regra)

            if "produto do dia" in regra:
                executar_tab3.validade = datetime(executar_tab3.agora.year, executar_tab3.agora.month, executar_tab3.agora.day, 23, 59, 59, tzinfo=TZ)
            elif dias:
                executar_tab3.validade = executar_tab3.agora + timedelta(days=int(dias.group(1)))
            elif horas:
                executar_tab3.validade = executar_tab3.agora + timedelta(hours=int(horas.group(1)))

            dt_forn = parse_data_forn(executar_tab3.val_for)
            if dt_forn and executar_tab3.validade and dt_forn < executar_tab3.validade:
                executar_tab3.validade = dt_forn
                executar_tab3.obs = (executar_tab3.obs or "")
                if "Fornecedor limitou validade" not in executar_tab3.obs:
                    executar_tab3.obs = (executar_tab3.obs + " | Fornecedor limitou validade").strip(" |")
            elif dt_forn and not executar_tab3.validade:
                executar_tab3.validade = dt_forn

            st.session_state.foto_haccp_pronto = bool(executar_tab3.validade and (executar_tab3.lote or "").strip())

    else:
        st.info("Tira ou carrega uma foto para começar.")

    # -------------------------
    # RESULTADO + PRINT (NÃO depende de img_file / pil_img)
    # -------------------------
    if st.session_state.foto_haccp_pronto and executar_tab3.validade and (executar_tab3.lote or "").strip():
        bloco = executar_tab3.texto_vizualizar_hccp()
        st.success("Validade gerada com sucesso (por foto)!")
        st.code(bloco, language="text")

        imp_id = st.selectbox(
            "Impressora Zebra",
            options=list(ZEBRAS.keys()),
            format_func=lambda k: f"{k} - {ZEBRAS[k]['nome']}",
            key="zebra_select_tab3",
        )

        if st.button("Imprimir etiqueta", key="btn_print_tab3"):
            zebra = ZEBRAS[imp_id]
            zpl = build_haccp_zpl(executar_tab3, dpi=zebra["dpi"])

            try:
                send_zpl_to_zebra(zebra["ip"], zpl)
                st.session_state.print_feedback = {
                    "kind": "success",
                    "msg": "Etiqueta enviada para a impressora com sucesso ✅"
                }
            except ConnectionError as e:
                st.session_state.print_feedback = {
                    "kind": "warning",
                    "msg": f"⚠️ Não foi possível imprimir. Motivo: {str(e)}"
                }
            except Exception as e:
                st.session_state.print_feedback = {
                    "kind": "error",
                    "msg": "❌ Erro inesperado ao tentar imprimir."
                }
                st.exception(e)

        st.download_button(
            "Baixar como TXT",
            data=bloco.encode("utf-8"),
            file_name=f"HCCP_{executar_tab3.nome[:20].replace(' ', '_')}_{executar_tab3.lote}_{executar_tab3.val_for}.txt",
            mime="text/plain",
            key="dl_txt_tab3",
        )
