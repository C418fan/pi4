import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Markowitz import obter_dados_ativos, calcular_retornos, calcular_variancia, calcular_retorno_portfolio, calcular_sharpe_ratio, otimizar_portfolio, otimizar_sharpe_ratio, fronteira_eficiente

# Variáveis de estado
if 'retorno_esperado_benchmark' not in st.session_state:
    st.session_state.retorno_esperado_benchmark = None
if 'variancia_benchmark' not in st.session_state:
    st.session_state.variancia_benchmark = None
if 'retorno_esperado_ativos' not in st.session_state:
    st.session_state.retorno_esperado_ativos = None
if 'matriz_cov' not in st.session_state:
    st.session_state.matriz_cov = None
if 'retornos_alvo' not in st.session_state:
    st.session_state.retornos_alvo = None
if 'volatilidades' not in st.session_state:
    st.session_state.volatilidades = None
if 'pesos' not in st.session_state:
    st.session_state.pesos = None
if 'pesos_max_sharpe' not in st.session_state:
    st.session_state.pesos_max_sharpe = None
if 'ponto_selecionado' not in st.session_state:
    st.session_state.ponto_selecionado = None
if 'tickers_lista' not in st.session_state:
    st.session_state.tickers_lista = None
if 'dados_ativos' not in st.session_state:
    st.session_state.dados_ativos = None
if 'dados_benchmark' not in st.session_state:
    st.session_state.dados_benchmark = None

# Layout
sidebar = st.sidebar
col1, col2 = st.columns(2)

# Sidebar
with sidebar:
    with st.form("inputs"):
        tickers_input = st.text_input('Tickers (ex: PETR4.SA,VALE3.SA)', 'PETR4.SA,VALE3.SA')
        benchmark_input = st.text_input('Benchmark (ex: ^BVSP)', '^BVSP')
        taxa_livre_risco = st.number_input('Taxa livre de risco (%)', min_value=0.0, value=4.0) / 100
        data_inicial = st.date_input("Data inicial", datetime(2020, 1, 1))
        data_final = st.date_input("Data final", datetime.now())
        
        submitted = st.form_submit_button("Calcular")
        
    if submitted:
        try:
            with st.spinner('Obtendo dados...'):
                dados_ativos, dados_benchmark = obter_dados_ativos(
                    tickers_input,
                    benchmark_input,
                    start=data_inicial,
                    end=data_final
                )
                
                retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark = calcular_retornos((dados_ativos, dados_benchmark))
                retornos_alvo, volatilidades, pesos = fronteira_eficiente(retorno_esperado_ativos, matriz_cov)
                pesos_max_sharpe = otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco)
                
                # Armazenar no estado
                st.session_state.update({
                    'dados_ativos': dados_ativos,
                    'dados_benchmark': dados_benchmark,
                    'retorno_esperado_benchmark': retorno_esperado_benchmark,
                    'variancia_benchmark': variancia_benchmark,
                    'matriz_cov': matriz_cov,
                    'retorno_esperado_ativos': retorno_esperado_ativos,
                    'retornos_alvo': retornos_alvo,
                    'volatilidades': volatilidades,
                    'pesos': pesos,
                    'pesos_max_sharpe': pesos_max_sharpe,
                    'tickers_lista': [t.strip() for t in tickers_input.split(',')]
                })
                
            st.success('Análise concluída!')
        except Exception as e:
            st.error(f"Erro: {str(e)}")

# Coluna 1 - Fronteira Eficiente
with col1:
    if st.session_state.retornos_alvo is not None:
        st.subheader('FRONTEIRA EFICIENTE')
        
        idx_min_vol = np.argmin(st.session_state.volatilidades)
        ponto_index = st.slider('Selecione um ponto', 0, len(st.session_state.volatilidades)-1, idx_min_vol)
        st.session_state.ponto_selecionado = ponto_index
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.volatilidades,
            y=st.session_state.retornos_alvo*100,
            mode='lines+markers',
            name='Fronteira Eficiente',
            line=dict(color='royalblue')
        ))
        
        # Adicionar pontos importantes
        fig.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[ponto_index]],
            y=[st.session_state.retornos_alvo[ponto_index]*100],
            mode='markers',
            name='Ponto Selecionado',
            marker=dict(color='blue', size=12)
        ))
        
        fig.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[idx_min_vol]], 
            y=[st.session_state.retornos_alvo[idx_min_vol]*100], 
            mode='markers', 
            name='Mínima Variância', 
            marker=dict(color='green', size=10, symbol='diamond')
        ))
        
        retorno_max_sharpe = calcular_retorno_portfolio(st.session_state.pesos_max_sharpe, st.session_state.retorno_esperado_ativos)
        volatilidade_max_sharpe = np.sqrt(calcular_variancia(st.session_state.pesos_max_sharpe, st.session_state.matriz_cov))
        
        fig.add_trace(go.Scatter(
            x=[volatilidade_max_sharpe], 
            y=[retorno_max_sharpe*100], 
            mode='markers', 
            name='Máximo Sharpe', 
            marker=dict(color='red', size=10, symbol='star')
        ))
        
        fig.update_layout(
            title='Fronteira Eficiente de Markowitz',
            xaxis_title='Volatilidade Anualizada (%)',
            yaxis_title='Retorno Esperado Anualizado (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# Coluna 2 - Composição e Performance
with col2:
    if st.session_state.pesos is not None:
        st.subheader('COMPOSIÇÃO DO PORTFÓLIO')
        
        pesos_mostrar = st.session_state.pesos[st.session_state.ponto_selecionado]
        pesos_percentual = pesos_mostrar * 100
        
        # Gráfico de pizza
        fig_pie = go.Figure(go.Pie(
            labels=st.session_state.tickers_lista,
            values=pesos_percentual,
            textinfo='label+percent',
            hole=0.3,
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96'])
        ))
        fig_pie.update_layout(title_text='Distribuição dos Pesos (%)')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # ==============================================
        # NOVO GRÁFICO DE LINHAS - RENTABILIDADE HISTÓRICA
        # ==============================================
        st.subheader('RENTABILIDADE HISTÓRICA')
        
        # Calcular retorno acumulado em %
        retornos_acum = (1 + st.session_state.dados_ativos.pct_change()).cumprod() * 100
        retorno_bench = (1 + st.session_state.dados_benchmark.pct_change()).cumprod() * 100
        
        fig_rent = go.Figure()
        
        # Adicionar cada ativo
        for ticker in retornos_acum.columns:
            fig_rent.add_trace(go.Scatter(
                x=retornos_acum.index,
                y=retornos_acum[ticker],
                mode='lines',
                name=ticker,
                hovertemplate="<b>%{fullData.name}</b><br>Data: %{x|%d/%m/%Y}<br>Retorno: %{y:.2f}%<extra></extra>"
            ))
        
        # Adicionar benchmark
        fig_rent.add_trace(go.Scatter(
            x=retorno_bench.index,
            y=retorno_bench,
            mode='lines',
            name=f'Benchmark ({benchmark_input})',
            line=dict(color='black', dash='dash'),
            hovertemplate="Benchmark: %{y:.2f}%"
        ))
        
        # Configurações do layout
        fig_rent.update_layout(
            title="Evolução do Retorno Acumulado",
            xaxis_title="Data",
            yaxis_title="Retorno Acumulado (%)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_rent, use_container_width=True)
