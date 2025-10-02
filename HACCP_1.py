# ============================
# HCCP – Validades (Streamlit)
# ============================

# --- IMPORTS ---
import re
from datetime import datetime, timedelta, time
import os

import pandas as pd
import streamlit as st

def parse_data_forn(txt: str) -> datetime | None:
    if not txt:
        return None
    txt = txt.strip()
    fmts = ["%d/%m/%Y %H:%M", "%d/%m/%Y", "%d/%m/%y"]
    for f in fmts:
        try:
            dt = datetime.strptime(txt, f)
            if f in ("%d/%m/%Y", "%d/%m/%y"):
                dt = datetime.combine(dt.date(), time(23, 59, 59))
            return dt
        except ValueError:
            continue
    return None


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

        # Botão de imprimir
        if st.button("Imprimir etiqueta", key=f"print_consulta_{hash(obj.nome)}"):
            # st.toast exibe uma mensagem temporária
            st.toast("Impressão ainda não está disponível. Em breve! 🖨️")

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
        executar.agora = datetime.now()

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
                            executar.agora.year, executar.agora.month, executar.agora.day, 23, 59, 59
                        )
                    elif dias:
                        executar.validade = executar.agora + timedelta(days=int(dias.group(1)))
                    elif horas:
                        executar.validade = executar.agora + timedelta(hours=int(horas.group(1)))
                    # Se não cair em nenhum caso, “validade” permanece None
                    break

            dt_forn = parse_data_forn(executar.val_for)
            if dt_forn and executar.val_for:
                if dt_forn < executar.validade:
                    executar.validade = dt_forn


            # 5) Exibição do bloco + opção de download
        if getattr(executar, "validade", None) is not None and (executar.lote or "").strip():
            bloco = executar.texto_vizualizar_hccp()
            st.success("Validade gerada com sucesso!")
            st.code(bloco, language="text")

            # Baixar como TXT (nome do arquivo inclui produto + lote + val_for)
            st.download_button(
                "Baixar como TXT",
                data=bloco.encode("utf-8"),
                file_name=f"HCCP_{executar.nome[:20].replace(' ', '_')}_{executar.lote}_{executar.val_for}.txt",
                mime="text/plain",
            )
            if st.button("Imprimir etiqueta", key=f"print_consulta_{executar.nome}"):
                st.toast("Impressão ainda não está disponível. Em breve! 🖨️")


            elif not (executar.lote or "").strip():
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
if "executar" not in st.session_state:
    st.session_state.executar = Produto()   # rascunho “vazio”
executar = st.session_state.executar        # atalho local para escrever menos


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

tab1, tab2 = st.tabs(["Consultar validade", "Gerar validade HACCP"])

with tab1:
    Produto.pesquisart(produtos)

with tab2:
    Produto.hccp_streamlit(executar, produtos)
