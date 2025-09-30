# Validades HCCP dos Frescos (Streamlit)

Aplicação web para calcular a validade HCCP de produtos “expostos”:
- Regras: **produto do dia**, **X dias**, **X horas/h**.
- Saída no mesmo formato do console + botão para **download TXT**.

## Requisitos
- Python 3.9+
- Dependências: ver `requirements.txt` (ex.: `streamlit>=1.33`, `tzdata>=2024.1`)

## Como rodar localmente
```bash
pip install -r requirements.txt
streamlit run hccp_app.py
