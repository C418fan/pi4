# NOVO BLOCO - Gráfico de rentabilidade acumulada
if st.session_state.dados_ativos is not None and st.session_state.dados_benchmark is not None:
    st.subheader('RENTABILIDADE DAS AÇÕES AO LONGO DO TEMPO')

    retornos_acumulados = (1 + st.session_state.dados_ativos.pct_change()).cumprod()
    retornos_acumulados_benchmark = (1 + st.session_state.dados_benchmark.pct_change()).cumprod()

    fig_ret_acumulada = go.Figure()

    for ticker in st.session_state.tickers_lista:
        fig_ret_acumulada.add_trace(go.Scatter(
            x=retornos_acumulados.index,
            y=retornos_acumulados[ticker],
            mode='lines',
            name=ticker,
            line=dict(color=st.session_state.cores_por_ticker.get(ticker, None))
        ))

    fig_ret_acumulada.add_trace(go.Scatter(
        x=retornos_acumulados_benchmark.index,
        y=retornos_acumulados_benchmark.iloc[:, 0],
        mode='lines',
        name=f'Benchmark ({benchmark_input})',
        line=dict(color='black', dash='dash')
    ))

    fig_ret_acumulada.update_layout(
        title='Rentabilidade Acumulada (Escala Log)',
        xaxis_title='Data',
        yaxis_title='Crescimento do Capital (x vezes)',
        yaxis_type='log',
        hovermode='x unified'
    )

    st.plotly_chart(fig_ret_acumulada, use_container_width=True)
