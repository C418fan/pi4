import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Markowitz import obter_dados_ativos, calcular_retornos, calcular_variancia, calcular_retorno_portfolio, calcular_sharpe_ratio, otimizar_portfolio, otimizar_sharpe_ratio, fronteira_eficiente, calcular_retorno_acumulado

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
        
        # Gráfico de performance histórica
        st.subheader('PERFORMANCE HISTÓRICA')
        ret_ativos, ret_bench = calcular_retorno_acumulado(
            (st.session_state.dados_ativos, st.session_state.dados_benchmark)
        )
        
        fig_hist = go.Figure()
        for ativo in ret_ativos.columns:
            fig_hist.add_trace(go.Scatter(
                x=ret_ativos.index,
                y=ret_ativos[ativo]*100,
                mode='lines',
                name=ativo
            ))
        
        fig_hist.add_trace(go.Scatter(
            x=ret_bench.index,
            y=ret_bench*100,
            mode='lines',
            name=f'Benchmark ({benchmark_input})',
            line=dict(color='black', dash='dash')
        ))
        
        fig_hist.update_layout(
            title='Retorno Acumulado (%)',
            xaxis_title='Data',
            yaxis_title='Retorno (%)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with sidebar:
    with st.form("inputs"):
        tickers_input = st.text_input('Tickers (ex: PETR4.SA,VALE3.SA,ITUB4.SA)', 'PETR4.SA,VALE3.SA')
        benchmark_input = st.text_input('Benchmark (ex: ^BVSP para Ibovespa)', '^BVSP')
        [... outros inputs ...]
        
    if submitted:
        try:
            with st.spinner('Obtendo dados...'):
                # Verifica tickers antes de baixar
                if not tickers_input or not benchmark_input:
                    raise ValueError("Preencha todos os campos de tickers")
                
                dados_ativos, dados_benchmark = obter_dados_ativos(
                    tickers=tickers_input,
                    benchmark=benchmark_input,
                    start=data_inicial,
                    end=data_final
                )
                
                # Verifica se obteve dados
                if dados_ativos.empty or dados_benchmark.empty:
                    raise ValueError("Nenhum dado encontrado - verifique tickers e período")
                
                [... restante do processamento ...]
                
        except Exception as e:
            st.error(f"""
            Falha ao obter dados:
            {str(e)}
            
            Soluções:
            1. Verifique os tickers (ex: PETR4.SA para Petrobras)
            2. Confira o benchmark (ex: ^BVSP para Ibovespa)
            3. Tente um período diferente
            4. Tickers internacionais precisam de sufixo (ex: AAPL para Apple)
            """)
            st.stop()
