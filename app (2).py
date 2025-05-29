import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Markowitz import obter_dados_ativos, calcular_retornos, calcular_variancia, calcular_retorno_portfolio, calcular_sharpe_ratio, otimizar_portfolio, otimizar_sharpe_ratio, fronteira_eficiente

# Inicialização de variáveis de estado
for key in [
    'retorno_esperado_benchmark', 'variancia_benchmark', 'retorno_esperado_ativos',
    'matriz_cov', 'retornos_alvo', 'volatilidades', 'pesos', 'pesos_max_sharpe',
    'ponto_selecionado', 'tickers_lista', 'dados_ativos', 'dados_benchmark',
    'cores_por_ticker'
]:
    if key not in st.session_state:
        st.session_state[key] = None

#Layout
sidebar = st.sidebar
col1 = st.container()
col2 = st.container()

with sidebar:
    with st.form("inputs"):
        tickers_input = st.text_input('Adicione os tickers separados por vírgula')
        benchmark_input = st.text_input('Adicione o benchmark')
        taxa_livre_risco = st.number_input('Adicione a taxa livre de risco', value=0.1)
        data_inicial = st.date_input("Data inicial", min_value=datetime(1988, 1, 1), max_value=datetime.now().date())
        data_final = st.date_input("Data final", min_value=datetime(1988, 1, 1), max_value=datetime.now().date())
        submitted = st.form_submit_button("Confirmar")

    if submitted and tickers_input and benchmark_input and data_inicial and data_final:
        try:
            dados_ativos, dados_benchmark = obter_dados_ativos(
                tickers_input, benchmark_input, start=data_inicial, end=data_final
            )
            retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark = calcular_retornos((dados_ativos, dados_benchmark))
            retornos_alvo, volatilidades, pesos = fronteira_eficiente(retorno_esperado_ativos, matriz_cov)
            pesos_max_sharpe = otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco)

            st.session_state.dados_ativos = dados_ativos
            st.session_state.dados_benchmark = dados_benchmark
            st.session_state.retorno_esperado_benchmark = retorno_esperado_benchmark
            st.session_state.variancia_benchmark = variancia_benchmark
            st.session_state.matriz_cov = matriz_cov
            st.session_state.retorno_esperado_ativos = retorno_esperado_ativos
            st.session_state.retornos_alvo = retornos_alvo
            st.session_state.volatilidades = volatilidades
            st.session_state.pesos = pesos
            st.session_state.pesos_max_sharpe = pesos_max_sharpe
            st.session_state.tickers_lista = tickers_input.replace(' ', '').split(',')

            # Definir cores padronizadas por ticker
            cores_tickers = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            st.session_state.cores_por_ticker = {
                ticker: cor for ticker, cor in zip(st.session_state.tickers_lista, cores_tickers)
            }

            st.success('Dados obtidos com sucesso!')
        except Exception as e:
            st.error(f'Erro ao obter dados: {str(e)}')

with col1:
    if st.session_state.retornos_alvo is not None:
        st.subheader('FRONTEIRA EFICIENTE')
        volatilidades_series = pd.Series(st.session_state.volatilidades)
        idx_min_vol = volatilidades_series.argmin()
        ponto_index = st.slider('Selecione um ponto na fronteira eficiente', 
                                0, len(st.session_state.volatilidades)-1, idx_min_vol)
        st.session_state.ponto_selecionado = ponto_index

        fig_fronteira = go.Figure()

        fig_fronteira.add_trace(go.Scatter(
            x=st.session_state.volatilidades,
            y=np.array(st.session_state.retornos_alvo)*100,
            mode='lines+markers',
            name='Fronteira Eficiente'
        ))

        fig_fronteira.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[ponto_index]],
            y=[st.session_state.retornos_alvo[ponto_index]*100],
            mode='markers',
            name='Ponto Selecionado',
            marker=dict(color='blue', size=15)
        ))

        fig_fronteira.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[idx_min_vol]],
            y=[st.session_state.retornos_alvo[idx_min_vol]*100],
            mode='markers',
            name='Portfólio de Mínima Variância',
            marker=dict(color='green', size=10)
        ))

        retorno_max_sharpe = calcular_retorno_portfolio(st.session_state.pesos_max_sharpe, st.session_state.retorno_esperado_ativos)
        volatilidade_max_sharpe = np.sqrt(calcular_variancia(st.session_state.pesos_max_sharpe, st.session_state.matriz_cov))
        fig_fronteira.add_trace(go.Scatter(
            x=[volatilidade_max_sharpe],
            y=[retorno_max_sharpe*100],
            mode='markers',
            name='Portfólio de Máxima Sharpe',
            marker=dict(color='red', size=10)
        ))

        volatilidade_benchmark = np.sqrt(st.session_state.variancia_benchmark[0])
        fig_fronteira.add_trace(go.Scatter(
            x=[volatilidade_benchmark],
            y=[st.session_state.retorno_esperado_benchmark[0]*100],
            mode='markers',
            name='Benchmark',
            marker=dict(color='yellow', size=10)
        ))

        fig_fronteira.update_layout(
            title='Fronteira Eficiente',
            xaxis_title='Volatilidade',
            yaxis_title='Retorno Esperado (%)'
        )

        st.plotly_chart(fig_fronteira, use_container_width=True)

with col2:
    if st.session_state.pesos is not None and st.session_state.ponto_selecionado is not None:
        st.subheader('PESOS DO PORTFÓLIO SELECIONADO')

        pesos_mostrar = st.session_state.pesos[st.session_state.ponto_selecionado]
        pesos_percentual = pesos_mostrar * 100
        indices_significativos = pesos_percentual >= 1.0

        if any(~indices_significativos):
            labels = []
            valores = []
            for ticker, peso, significativo in zip(st.session_state.tickers_lista, pesos_percentual, indices_significativos):
                if significativo:
                    labels.append(ticker)
                    valores.append(peso)
            labels.append('Outros')
            valores.append(sum(pesos_percentual[~indices_significativos]))
        else:
            labels = st.session_state.tickers_lista
            valores = pesos_percentual

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=labels,
            values=valores,
            textinfo='label+percent',
            hovertemplate="Ticker: %{label}<br>Peso: %{value:.1f}%<extra></extra>",
            sort=True,
            direction='clockwise',
            pull=[0.1 if v >= 1 else 0 for v in valores],
            marker=dict(colors=[st.session_state.cores_por_ticker.get(label, '#CCCCCC') for label in labels])
        ))

        fig_pie.update_layout(
            title='Distribuição dos Pesos (%)',
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)



# Gráfico final: Rentabilidade acumulada com base 100
if st.session_state.dados_ativos is not None and st.session_state.dados_benchmark is not None:
    st.subheader('RENTABILIDADE DAS AÇÕES AO LONGO DO TEMPO')

    # Normalizar os preços para começarem em 100
    precos_normalizados = st.session_state.dados_ativos / st.session_state.dados_ativos.iloc[0] * 100
    benchmark_normalizado = st.session_state.dados_benchmark / st.session_state.dados_benchmark.iloc[0] * 100

    fig_ret_base100 = go.Figure()

    for ticker in st.session_state.tickers_lista:
        fig_ret_base100.add_trace(go.Scatter(
            x=precos_normalizados.index,
            y=precos_normalizados[ticker],
            mode='lines',
            name=ticker,
            line=dict(color=st.session_state.get('cores_por_ticker', {}).get(ticker, None))
        ))



    fig_ret_base100.update_layout(
        title='Rentabilidade Acumulada com Base 100',
        xaxis_title='Data',
        hovermode='x unified'
    )

    st.plotly_chart(fig_ret_base100, use_container_width=True)

# NOVO GRÁFICO: Comparação entre o portfólio real e o IBOVESPA com base 100
if (
    st.session_state.dados_ativos is not None and 
    st.session_state.dados_benchmark is not None and 
    st.session_state.ponto_selecionado is not None
):
    st.subheader('COMPARAÇÃO: PORTFÓLIO REAL vs IBOVESPA (Base 100)')

    # Dados históricos
    dados_ativos = st.session_state.dados_ativos
    dados_benchmark = st.session_state.dados_benchmark.iloc[:, 0]  # Série do benchmark

    # Normalizar benchmark para base 100
    ibov_normalizado = dados_benchmark / dados_benchmark.iloc[0] * 100

    # Pesos do portfólio no ponto selecionado
    pesos = st.session_state.pesos[st.session_state.ponto_selecionado]

    # Normalizar ativos para base 100
    ativos_normalizados = dados_ativos / dados_ativos.iloc[0] * 100

    # Calcular índice do portfólio real ponderado
    portfolio_real = ativos_normalizados.dot(pesos)

    # Plotar gráfico
    fig_portfolio_vs_ibov = go.Figure()
    fig_portfolio_vs_ibov.add_trace(go.Scatter(
        x=portfolio_real.index,
        y=portfolio_real,
        mode='lines',
        name='Portfólio Real',
        line=dict(color='green')
    ))

    fig_portfolio_vs_ibov.add_trace(go.Scatter(
        x=ibov_normalizado.index,
        y=ibov_normalizado,
        mode='lines',
        name='IBOVESPA',
        line=dict(color='white', dash='dash')
    ))

    fig_portfolio_vs_ibov.update_layout(
        title='Comparação: Portfólio Real vs IBOVESPA (Base 100)',
        xaxis_title='Data',
        hovermode='x unified'
    )

    st.plotly_chart(fig_portfolio_vs_ibov, use_container_width=True)

