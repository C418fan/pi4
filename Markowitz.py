import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st  
from scipy.optimize import minimize
import plotly.graph_objects as go


def obter_dados_ativos(tickers_str, benchmark, start, end):
    tickers = tickers_str.replace(" ", "").split(",")

    try:
        # Baixar dados dos ativos com auto ajuste
        dados = yf.download(tickers + [benchmark], start=start, end=end, auto_adjust=True, progress=False)

        if dados.empty:
            raise ValueError("Nenhum dado retornado pelo Yahoo Finance.")

        # Extrair apenas a coluna 'Close' (funciona com múltiplos tickers e benchmark)
        if isinstance(dados.columns, pd.MultiIndex) and 'Close' in dados.columns.levels[0]:
            dados_close = dados['Close']
        else:
            raise ValueError("Coluna 'Close' não encontrada nos dados baixados.")

        # Separar benchmark dos ativos
        if benchmark not in dados_close.columns:
            raise ValueError(f"Benchmark '{benchmark}' não encontrado nos dados.")
        dados_benchmark = dados_close[[benchmark]]
        dados_ativos = dados_close.drop(columns=[benchmark], errors='ignore')

        if dados_ativos.empty:
            raise ValueError("Nenhum dado de fechamento válido encontrado para os ativos.")
        if dados_benchmark.empty:
            raise ValueError("Nenhum dado de fechamento válido encontrado para o benchmark.")

        return dados_ativos.dropna(), dados_benchmark.dropna()

    except Exception as e:
        raise Exception(f"Falha ao obter dados: - Tickers: {','.join(tickers)} - Benchmark: {benchmark} Causa: {str(e)}")

def calcular_retornos(dados):
    """
    Calcula retornos compostos dos ativos e do índice e estatísticas necessárias.
    """
    retornos_ativos = dados[0].pct_change().dropna()
    retornos_benchmark = dados[1].pct_change().dropna()
    retorno_esperado_ativos = (retornos_ativos.mean() + 1) ** 252 - 1       
    retorno_esperado_benchmark = (retornos_benchmark.mean() + 1) ** 252 - 1
    variancia_benchmark = retornos_benchmark.var()
    matriz_cov = retornos_ativos.cov() * 252
    return retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark

def calcular_variancia(pesos, matriz_cov):
    """
    Calcula a variância do portfólio.
    """
    return np.dot(pesos.T, np.dot(matriz_cov, pesos))

def calcular_retorno_portfolio(pesos, retorno_esperado_ativos):
    """
    Calcula o retorno esperado do portfólio.
    """
    return np.dot(pesos, retorno_esperado_ativos)

def calcular_sharpe_ratio(pesos, retorno_esperado_ativos, matriz_cov, taxa_livre_risco):
    """
    Calcula o Índice de Sharpe do portfólio.
    """
    retorno_portfolio = calcular_retorno_portfolio(pesos, retorno_esperado_ativos)
    volatilidade_portfolio = np.sqrt(calcular_variancia(pesos, matriz_cov))
    return (retorno_portfolio - taxa_livre_risco) / volatilidade_portfolio

def otimizar_portfolio(retorno_alvo, retorno_esperado_ativos, matriz_cov):
    """
    Otimiza os pesos para mínimo risco dado um retorno alvo.
    """
    n_ativos = len(retorno_esperado_ativos)
    args = (matriz_cov,)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: calcular_retorno_portfolio(x, retorno_esperado_ativos) - retorno_alvo}
    )
    bounds = tuple((0, 1) for _ in range(n_ativos))
    resultado = minimize(calcular_variancia, n_ativos * [1/n_ativos], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return resultado.x

def otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco):
    """
    Otimiza os pesos para máximo Índice de Sharpe.
    """
    n_ativos = len(retorno_esperado_ativos)
    args = (retorno_esperado_ativos, matriz_cov, taxa_livre_risco)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_ativos))
    resultado = minimize(lambda x: -calcular_sharpe_ratio(x, *args), n_ativos * [1/n_ativos],
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return resultado.x

def fronteira_eficiente(retorno_esperado_ativos, matriz_cov, n_pontos=100):
    """
    Gera pontos da fronteira eficiente.
    """
    retornos_alvo = np.linspace(retorno_esperado_ativos.min(), retorno_esperado_ativos.max(), n_pontos)
    volatilidades = []
    pesos = []

    for retorno in retornos_alvo:
        peso = otimizar_portfolio(retorno, retorno_esperado_ativos, matriz_cov)
        pesos.append(peso)
        volatilidades.append(np.sqrt(calcular_variancia(peso, matriz_cov)))

    return retornos_alvo, volatilidades, np.array(pesos)

def plotar_fronteira_interativa(retornos_alvo, volatilidades, retorno_esperado_ativos, matriz_cov, pesos_max_sharpe, taxa_livre_risco):
    """
    Plota a fronteira eficiente de forma interativa com Plotly.
    """
    # Cálculo do portfólio de máxima Sharpe
    retorno_max_sharpe = calcular_retorno_portfolio(pesos_max_sharpe, retorno_esperado_ativos)
    volatilidade_max_sharpe = np.sqrt(calcular_variancia(pesos_max_sharpe, matriz_cov))

    # Criação do gráfico
    fig = go.Figure()

    # Fronteira Eficiente
    fig.add_trace(go.Scatter(
        x=volatilidades,
        y=retornos_alvo,
        mode='lines+markers',
        name='Fronteira Eficiente',
        line=dict(color='blue', width=2),
        marker=dict(size=5),
        hovertemplate="<b>Volatilidade:</b> %{x:.2%}<br><b>Retorno:</b> %{y:.2%}<extra></extra>"
    ))

    # Portfólio de Mínima Variância
    idx_min_var = np.argmin(volatilidades)
    fig.add_trace(go.Scatter(
        x=[volatilidades[idx_min_var]],
        y=[retornos_alvo[idx_min_var]],
        mode='markers',
        name='Mínima Variância',
        marker=dict(color='green', size=12, symbol='star'),
        hovertemplate="<b>Mínima Variância</b><br>Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
    ))

    # benchmark
    fig.add_trace(go.Scatter(
        x=[variancia_benchmark],
        y=[retorno_esperado_benchmark],
        mode='markers',
        name='benchmark',
        marker=dict(color='black', size=12, symbol='star'),
        hovertemplate="<b>benchmark</b><br>Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
    ))

    # Portfólio de Máximo Sharpe
    fig.add_trace(go.Scatter(
        x=[volatilidade_max_sharpe],
        y=[retorno_max_sharpe],
        mode='markers',
        name='Máximo Sharpe',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate="<b>Máximo Sharpe</b><br>Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
    ))

    # Layout do gráfico
    fig.update_layout(
        title="Fronteira Eficiente de Markowitz",
        xaxis_title="Volatilidade (Risco)",
        yaxis_title="Retorno anualizado Esperado",
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Exibir o gráfico
    fig.show()

if __name__ == "__main__":

    # Entrada de dados do usuário
    tickers = input("Digite os tickers dos ativos (separados por vírgula)\n").upper().strip('')
    benchmark = input("Digite o benchmark\n").upper().strip('')
    start = input("Digite a data de inicio do periodo (Formato YYYY-MM-DD): ").upper().strip('')
    end = input("Digite a data de fim do periodo (Formato YYYY-MM-DD): ").upper().strip('')
    taxa_livre_risco = float(input("Digite a taxa livre de risco anual (em decimal, ex: 0.03 para 3%): "))

    # Obter e processar dados
    dados = obter_dados_ativos(tickers, benchmark)
    retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark = calcular_retornos(dados)

    # Calcular fronteira eficiente
    retornos_alvo, volatilidades, pesos = fronteira_eficiente(retorno_esperado_ativos, matriz_cov)

    # Otimizar portfólio de máxima Sharpe
    pesos_max_sharpe = otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco)

    # Plotar gráfico interativo
    plotar_fronteira_interativa(retornos_alvo, volatilidades, retorno_esperado_ativos, matriz_cov, pesos_max_sharpe, taxa_livre_risco) 
    
