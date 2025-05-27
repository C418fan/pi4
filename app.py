import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Markowitz import obter_dados_ativos, calcular_retornos, calcular_variancia, calcular_retorno_portfolio, calcular_sharpe_ratio, otimizar_portfolio, otimizar_sharpe_ratio, fronteira_eficiente

# Variáveis de estado para armazenar os dados
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

# Criando duas colunas: uma para o sidebar e outra para os dados
sidebar = st.sidebar
col1, col2 = st.columns(2)

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
                # Verificação preliminar
                if not tickers_input or not benchmark_input:
                    raise ValueError("Preencha todos os campos de tickers")
                
                # Obter dados com tratamento de erro
                dados_ativos, dados_benchmark = obter_dados_ativos(
                    tickers_input,
                    benchmark_input,
                    start=data_inicial,
                    end=data_final
                )
                
                # Verificação adicional de dados vazios
                if dados_ativos.empty or dados_benchmark.empty:
                    raise ValueError("Nenhum dado encontrado - verifique tickers e período")
                
                # Calcula os retornos esperados e a covariância
                retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark = calcular_retornos((dados_ativos, dados_benchmark))
                
                # Calcular fronteira eficiente
                retornos_alvo, volatilidades, pesos = fronteira_eficiente(retorno_esperado_ativos, matriz_cov)

                # Otimizar portfólio de máxima Sharpe
                pesos_max_sharpe = otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco)

                # Armazenando os dados no session_state
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
                st.session_state.tickers_lista = [t.strip() for t in tickers_input.split(',')]
                
            st.success('Análise concluída com sucesso!')
        except Exception as e:
            st.error(f"""
            Erro ao processar dados:
            {str(e)}
            
            Soluções possíveis:
            1. Verifique se os tickers estão corretos (ex: PETR4.SA para ações brasileiras)
            2. Confira se o benchmark existe (ex: ^BVSP para Ibovespa)
            3. Tente um período de tempo diferente
            4. Para ações internacionais, use o código correto (ex: AAPL)
            """)

with col1:
    if st.session_state.retornos_alvo is not None:
        st.subheader('FRONTEIRA EFICIENTE')
        
        # Encontrando o portfólio de mínima variância
        idx_min_vol = np.argmin(st.session_state.volatilidades)
        ponto_index = st.slider('Selecione um ponto na fronteira eficiente', 
                              0, 
                              len(st.session_state.volatilidades)-1, 
                              idx_min_vol)
        st.session_state.ponto_selecionado = ponto_index
        
        fig = go.Figure()

        # Adicionando a linha da fronteira eficiente
        fig.add_trace(go.Scatter(
            x=st.session_state.volatilidades,
            y=np.array(st.session_state.retornos_alvo)*100,
            mode='lines+markers',
            name='Fronteira Eficiente',
            line=dict(color='royalblue', width=2)
        ))

        # Ponto selecionado
        fig.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[ponto_index]],
            y=[st.session_state.retornos_alvo[ponto_index]*100],
            mode='markers',
            name='Ponto Selecionado',
            marker=dict(color='blue', size=12)
        ))

        # Portfólio de mínima variância
        fig.add_trace(go.Scatter(
            x=[st.session_state.volatilidades[idx_min_vol]], 
            y=[st.session_state.retornos_alvo[idx_min_vol]*100], 
            mode='markers', 
            name='Portfólio de Mínima Variância', 
            marker=dict(color='green', size=10, symbol='diamond')
        ))

        # Portfólio de máxima Sharpe
        retorno_max_sharpe = calcular_retorno_portfolio(st.session_state.pesos_max_sharpe, st.session_state.retorno_esperado_ativos)
        volatilidade_max_sharpe = np.sqrt(calcular_variancia(st.session_state.pesos_max_sharpe, st.session_state.matriz_cov))
        
        fig.add_trace(go.Scatter(
            x=[volatilidade_max_sharpe], 
            y=[retorno_max_sharpe*100], 
            mode='markers', 
            name='Portfólio de Máxima Sharpe', 
            marker=dict(color='red', size=10, symbol='star')
        ))

        # Benchmark
        volatilidade_benchmark = np.sqrt(st.session_state.variancia_benchmark)
        fig.add_trace(go.Scatter(
            x=[volatilidade_benchmark], 
            y=[st.session_state.retorno_esperado_benchmark*100], 
            mode='markers', 
            name='Benchmark', 
            marker=dict(color='gold', size=10, symbol='square')
        ))

        fig.update_layout(
            title='Fronteira Eficiente de Markowitz', 
            xaxis_title='Volatilidade Anualizada (%)', 
            yaxis_title='Retorno Esperado Anualizado (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if st.session_state.pesos is not None and st.session_state.ponto_selecionado is not None:
        st.subheader('COMPOSIÇÃO DO PORTFÓLIO')
        
        # Usando os pesos do ponto selecionado
        pesos_mostrar = st.session_state.pesos[st.session_state.ponto_selecionado]
        pesos_percentual = pesos_mostrar * 100
        
        # Gráfico de pizza
        fig_pie = go.Figure(go.Pie(
            labels=st.session_state.tickers_lista,
            values=pesos_percentual,
            textinfo='label+percent',
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Plotly),
            hovertemplate="<b>%{label}</b><br>Peso: %{percent}<extra></extra>"
        ))
        fig_pie.update_layout(
            title='Distribuição dos Pesos',
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # GRÁFICO DE RENTABILIDADE HISTÓRICA (NOVO)
        st.subheader('RENTABILIDADE HISTÓRICA')
        
        # Calcular retornos acumulados
        retornos_acumulados = (1 + st.session_state.dados_ativos.pct_change()).cumprod() * 100
        retorno_bench_acumulado = (1 + st.session_state.dados_benchmark.pct_change()).cumprod() * 100
        
        fig_rent = go.Figure()
        
        # Adicionar cada ativo
        for ticker in st.session_state.tickers_lista:
            fig_rent.add_trace(go.Scatter(
                x=retornos_acumulados.index,
                y=retornos_acumulados[ticker],
                mode='lines',
                name=ticker,
                hovertemplate="<b>%{fullData.name}</b><br>Data: %{x|%d/%m/%Y}<br>Retorno: %{y:.2f}%<extra></extra>"
            ))
        
        # Adicionar benchmark
        fig_rent.add_trace(go.Scatter(
            x=retorno_bench_acumulado.index,
            y=retorno_bench_acumulado,
            mode='lines',
            name=f'Benchmark ({benchmark_input})',
            line=dict(color='black', dash='dash'),
            hovertemplate="<b>Benchmark</b><br>%{y:.2f}%<extra></extra>"
        ))
        
        # Linha de referência (100% = valor inicial)
        fig_rent.add_hline(y=100, line_dash="dot", line_color="gray")
        
        fig_rent.update_layout(
            title='Evolução da Rentabilidade',
            xaxis_title='Data',
            yaxis_title='Retorno Acumulado (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_rent, use_container_width=True)
