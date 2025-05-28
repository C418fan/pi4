import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def obter_dados_ativos(tickers_str, benchmark, start, end):
    tickers = tickers_str.replace(" ", "").split(",")
    try:
        dados_ativos = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        dados_benchmark = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=True)

        # TRATAMENTO DE ERRO: verifica se há coluna 'Close'
        if isinstance(dados_ativos.columns, pd.MultiIndex):
            dados_ativos = dados_ativos['Close']
        elif 'Close' in dados_ativos.columns:
            dados_ativos = dados_ativos[['Close']]
        else:
            raise ValueError("Dados de ativos inválidos: coluna 'Close' não encontrada.")

        if isinstance(dados_benchmark.columns, pd.MultiIndex):
            dados_benchmark = dados_benchmark['Close']
        elif 'Close' in dados_benchmark.columns:
            dados_benchmark = dados_benchmark[['Close']]
        else:
            raise ValueError("Dados do benchmark inválidos: coluna 'Close' não encontrada.")

        return dados_ativos.dropna(), dados_benchmark.dropna()

    except Exception as e:
        raise Exception(f"Falha ao obter dados: - Tickers: {tickers_str} - Benchmark: {benchmark} Causa: {str(e)}")

        # Função para extrair preços com fallback
        def extrair_precos(df):
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.get_level_values(1):
                    return df.xs('Adj Close', axis=1, level=1).dropna()
                return df.xs('Close', axis=1, level=1).dropna()
            return df['Adj Close'].dropna() if 'Adj Close' in df.columns else df['Close'].dropna()

        return extrair_precos(dados), extrair_precos(bench)
        
    except Exception as e:
        raise ValueError(f"""
        Falha ao obter dados:
        - Tickers: {tickers}
        - Benchmark: {benchmark}
        Causa: {str(e)}
        
        Soluções:
        1. Verifique se os tickers existem (ex: PETR4.SA para Petrobras)
        2. Para ações internacionais, use o código correto (ex: AAPL)
        3. Tickers brasileiros precisam do sufixo .SA
        4. Verifique se o benchmark está correto (ex: ^BVSP para Ibovespa)
        """)
def calcular_retornos(dados):
    """Calcula retornos anuais e matriz de covariância"""
    retornos_ativos = dados[0].pct_change().dropna()
    retornos_bench = dados[1].pct_change().dropna()
    
    ret_anual_ativos = (1 + retornos_ativos.mean())**252 - 1
    ret_anual_bench = (1 + retornos_bench.mean())**252 - 1
    variancia_bench = retornos_bench.var()
    cov_matrix = retornos_ativos.cov() * 252
    
    return ret_anual_ativos, cov_matrix, ret_anual_bench, variancia_bench

def calcular_retorno_acumulado(dados):
    """Calcula retorno acumulado para o gráfico temporal"""
    return (1 + dados[0].pct_change()).cumprod(), (1 + dados[1].pct_change()).cumprod()

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
