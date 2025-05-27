import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go

def obter_dados_ativos(tickers, benchmark, start=None, end=None):
    """Obtém dados históricos de fechamento ajustado com tratamento de erros"""
    try:
        # Baixa dados dos ativos
        dados_ativos = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            progress=False,
            group_by='ticker'
        )
        
        # Verifica se os dados foram obtidos
        if dados_ativos.empty:
            raise ValueError(f"Não foi possível obter dados para os tickers: {tickers}")
            
        # Extrai coluna de fechamento ajustado
        if isinstance(dados_ativos.columns, pd.MultiIndex):
            adj_close = dados_ativos.xs('Adj Close', axis=1, level=1).dropna()
        else:
            adj_close = dados_ativos['Adj Close'].dropna()
        
        # Baixa dados do benchmark
        dados_bench = yf.download(
            tickers=benchmark,
            start=start,
            end=end,
            progress=False
        )
        
        if dados_bench.empty:
            raise ValueError(f"Não foi possível obter dados para o benchmark: {benchmark}")
            
        bench_close = dados_bench['Adj Close'].dropna()
        
        return adj_close, bench_close
        
    except Exception as e:
        error_msg = f"Erro ao obter dados: {str(e)}"
        if 'Adj Close' in str(e):
            error_msg = "Erro: Dados de fechamento ajustado não disponíveis. Verifique os tickers."
        raise ValueError(error_msg)

def calcular_retornos(dados):
    """Calcula retornos compostos e estatísticas necessárias"""
    retornos_ativos = dados[0].pct_change().dropna()
    retornos_benchmark = dados[1].pct_change().dropna()
    
    retorno_esperado_ativos = (retornos_ativos.mean() + 1) ** 252 - 1       
    retorno_esperado_benchmark = (retornos_benchmark.mean() + 1) ** 252 - 1
    variancia_benchmark = retornos_benchmark.var()
    matriz_cov = retornos_ativos.cov() * 252
    
    return retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark

def calcular_variancia(pesos, matriz_cov):
    """Calcula a variância do portfólio"""
    return np.dot(pesos.T, np.dot(matriz_cov, pesos))

def calcular_retorno_portfolio(pesos, retorno_esperado_ativos):
    """Calcula o retorno esperado do portfólio"""
    return np.dot(pesos, retorno_esperado_ativos)

def calcular_sharpe_ratio(pesos, retorno_esperado_ativos, matriz_cov, taxa_livre_risco):
    """Calcula o Índice de Sharpe"""
    retorno = calcular_retorno_portfolio(pesos, retorno_esperado_ativos)
    risco = np.sqrt(calcular_variancia(pesos, matriz_cov))
    return (retorno - taxa_livre_risco) / risco

def otimizar_portfolio(retorno_alvo, retorno_esperado_ativos, matriz_cov):
    """Otimiza os pesos para mínimo risco dado um retorno alvo"""
    n = len(retorno_esperado_ativos)
    args = (matriz_cov,)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: calcular_retorno_portfolio(x, retorno_esperado_ativos) - retorno_alvo}
    )
    bounds = tuple((0, 1) for _ in range(n))
    resultado = minimize(calcular_variancia, n * [1/n], args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return resultado.x

def otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco):
    """Otimiza os pesos para máximo Índice de Sharpe"""
    n = len(retorno_esperado_ativos)
    args = (retorno_esperado_ativos, matriz_cov, taxa_livre_risco)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    resultado = minimize(lambda x: -calcular_sharpe_ratio(x, *args), n * [1/n],
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return resultado.x

def fronteira_eficiente(retorno_esperado_ativos, matriz_cov, n_pontos=50):
    """Gera pontos da fronteira eficiente"""
    retornos = np.linspace(retorno_esperado_ativos.min(), retorno_esperado_ativos.max(), n_pontos)
    volatilidades = []
    pesos = []
    
    for r in retornos:
        w = otimizar_portfolio(r, retorno_esperado_ativos, matriz_cov)
        pesos.append(w)
        volatilidades.append(np.sqrt(calcular_variancia(w, matriz_cov)))
    
    return retornos, volatilidades, np.array(pesos)
