import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import datetime

# Configuración de la página
st.set_page_config(page_title="Análisis de Segmentación POS", page_icon="🏪", layout="wide")

# Función para cargar los datos
@st.cache_data
def cargar_datos():
    try:
        # Cargar el único dataset de segmentación
        df = pd.read_csv('df_Clustering_ICP.csv')
        
        # Asegurar que las columnas numéricas sean float
        numeric_cols = ['amount', 'avg_daily_savings_normalized', 'time_saving_coef_normalized', 
                      'matched_score_normalized', 'active_days_coef_normalized', 'dynamic_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculamos el valor de suscripción predicho basado en las métricas disponibles
        # La ponderación puede ajustarse según sea necesario
        df['predicted_subscription_value'] = 1000*(
            df['avg_daily_savings_normalized'] * 0.4 + 
            df['time_saving_coef_normalized'] * 0.2 + 
            df['matched_score_normalized'] * 0.2 + 
            df['active_days_coef_normalized'] * 0.2  
            #df['dynamic_score'] * 0.2
        ).fillna(0)
        
        # Categorizar amount en rangos para visualización
        try:
            df['amount_category'] = pd.qcut(
                df['amount'], q=3, labels=['Bajo', 'Medio', 'Alto']
            )
        except ValueError:
            # Alternativa si qcut falla
            df['amount_category'] = pd.cut(
                df['amount'], bins=3, labels=['Bajo', 'Medio', 'Alto']
            )
        
        # Categorizar valor predicho en rangos para visualización
        try:
            df['predicted_category'] = pd.qcut(
                df['predicted_subscription_value'], q=3, labels=['Bajo', 'Medio', 'Alto']
            )
        except ValueError:
            # Alternativa si qcut falla
            df['predicted_category'] = pd.cut(
                df['predicted_subscription_value'], bins=3, labels=['Bajo', 'Medio', 'Alto']
            )
        
        # Convertir categorías a string para evitar problemas
        df['amount_category'] = df['amount_category'].astype(str)
        df['predicted_category'] = df['predicted_category'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

# Título principal
st.title("🏪 Análisis de Segmentación de Puntos de Venta")
st.markdown("Esta aplicación permite segmentar puntos de venta según valor real (amount) y valor predicho.")

# Cargar datos
df = cargar_datos()

if not df.empty:
    # Información del dataset
    st.sidebar.header("Información del Dataset")
    st.sidebar.metric("Total Puntos de Venta", len(df))
    st.sidebar.metric("Valor Real Promedio", f"${df['amount'].mean():.2f}")
    st.sidebar.metric("Valor Predicho Promedio", f"{df['predicted_subscription_value'].mean():.2f}")
    
    # Filtros
    st.sidebar.header("Filtros")
    
    # Filtro por categoría de amount
    categorias_amount = ['Todas'] + sorted(df['amount_category'].unique().tolist())
    categoria_amount = st.sidebar.selectbox("Categoría de Valor Real", categorias_amount)
    
    # Filtro por categoría de predicción
    categorias_pred = ['Todas'] + sorted(df['predicted_category'].unique().tolist())
    categoria_pred = st.sidebar.selectbox("Categoría de Valor Predicho", categorias_pred)
    
    # Filtros de rango
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        min_amount, max_amount = float(df['amount'].min()), float(df['amount'].max())
        rango_amount = st.slider(
            "Rango de Valor Real", min_value=min_amount, max_value=max_amount, value=(min_amount, max_amount)
        )
    
    with col2:
        min_pred, max_pred = float(df['predicted_subscription_value'].min()), float(df['predicted_subscription_value'].max())
        rango_pred = st.slider(
            "Rango de Predicción", min_value=min_pred, max_value=max_pred, value=(min_pred, max_pred)
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if categoria_amount != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['amount_category'] == categoria_amount]
    if categoria_pred != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['predicted_category'] == categoria_pred]
    
    df_filtrado = df_filtrado[
        (df_filtrado['amount'] >= rango_amount[0]) & 
        (df_filtrado['amount'] <= rango_amount[1]) &
        (df_filtrado['predicted_subscription_value'] >= rango_pred[0]) & 
        (df_filtrado['predicted_subscription_value'] <= rango_pred[1])
    ]
    
    # Actualizar métricas según filtros
    st.sidebar.metric("Puntos Filtrados", len(df_filtrado))
    
    # Pestañas
    tab1, tab2, tab3 = st.tabs(["📊 Análisis", "🔍 Segmentación", "📋 Datos"])
    
    # PESTAÑA 1: ANÁLISIS
    with tab1:
        st.header("Análisis de Valores Reales vs. Predichos")
        
        if len(df_filtrado) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribución de amount
                fig = px.histogram(
                    df_filtrado, 
                    x='amount',
                    title="Distribución de Valores Reales",
                    color='amount_category',
                    marginal="box"
                )
                fig.update_layout(xaxis_title="Valor Real (amount)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de dispersión entre valor real y predicho
                fig = px.scatter(
                    df_filtrado,
                    x='predicted_subscription_value',
                    y='amount',
                    color='amount_category',
                    hover_name='id_pos',
                    trendline="ols",
                    title="Relación entre Valor Predicho y Valor Real"
                )
                fig.update_layout(
                    xaxis_title="Valor Predicho",
                    yaxis_title="Valor Real (amount)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribución de predicciones
                fig = px.histogram(
                    df_filtrado, 
                    x='predicted_subscription_value',
                    title="Distribución de Valores Predichos",
                    color='predicted_category',
                    marginal="box"
                )
                fig.update_layout(xaxis_title="Valor Predicho")
                st.plotly_chart(fig, use_container_width=True)
                
                # Matriz de correlación
                cols_correlacion = [
                    'amount', 'predicted_subscription_value', 'avg_daily_savings_normalized', 
                    'time_saving_coef_normalized', 'matched_score_normalized', 
                    'active_days_coef_normalized'
                ]
                corr_matrix = df_filtrado[cols_correlacion].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title="Matriz de Correlación entre Variables"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de contribución de variables a prediction
            st.subheader("Contribución de Variables al Valor Real")
            
            # Preparar datos para gráfico de importancia
            variables_importancia = [
                'avg_daily_savings_normalized', 'time_saving_coef_normalized',
                'matched_score_normalized', 'active_days_coef_normalized'#, 'dynamic_score'
            ]
            
            # Calcular correlación con amount para determinar importancia real
            correlaciones = []
            for var in variables_importancia:
                if var in df_filtrado.columns:
                    corr = df_filtrado[var].corr(df_filtrado['amount'])
                    correlaciones.append({'Variable': var, 'Correlación': abs(corr)})
            
            if correlaciones:
                df_corr = pd.DataFrame(correlaciones)
                df_corr = df_corr.sort_values('Correlación', ascending=False)
                
                fig = px.bar(
                    df_corr,
                    x='Variable',
                    y='Correlación',
                    title="Importancia de Variables (Correlación Absoluta con Valor Real)",
                    color='Correlación',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # PESTAÑA 2: SEGMENTACIÓN
    with tab2:
        st.header("Segmentación de Puntos de Venta")
        
        if len(df_filtrado) >= 5:  # Necesitamos al menos 5 puntos para segmentación
            st.write("Realiza una segmentación personalizada usando K-means.")
            
            # Seleccionar características para clusterización
            features_opciones = [
                'amount', 'predicted_subscription_value', 'avg_daily_savings_normalized', 
                'time_saving_coef_normalized', 'matched_score_normalized', 
                'active_days_coef_normalized', 'dynamic_score'
            ]
            
            features_seleccionadas = st.multiselect(
                "Seleccionar características para segmentación",
                features_opciones,
                default=['amount', 'predicted_subscription_value']
            )
            
            if features_seleccionadas:
                # Seleccionar número de clusters
                num_clusters = st.slider("Número de segmentos (clusters)", 2, 6, 3)
                
                # Preparar datos para K-means
                X = df_filtrado[features_seleccionadas].copy()
                X = X.fillna(X.mean())  # Manejar NaN
                
                # Normalizar
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Aplicar K-means
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    df_filtrado['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Etiquetar clusters
                    df_filtrado['etiqueta_cluster'] = df_filtrado['cluster'].apply(lambda x: f'Segmento {x+1}')
                    
                    # Visualizar resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if len(features_seleccionadas) >= 2:
                            # Determinar si necesitamos PCA (más de 2 características)
                            if len(features_seleccionadas) > 2:
                                st.info("Aplicando reducción dimensional (PCA) para visualizar datos multidimensionales en 2D")
                                
                                # Aplicar PCA para reducir a 2 dimensiones
                                pca = PCA(n_components=2)
                                pca_result = pca.fit_transform(X_scaled)
                                
                                # Crear un DataFrame con los resultados de PCA
                                pca_df = pd.DataFrame(data=pca_result, columns=['Componente 1', 'Componente 2'])
                                pca_df['etiqueta_cluster'] = df_filtrado['etiqueta_cluster'].values
                                pca_df['id_pos'] = df_filtrado['id_pos'].values
                                pca_df['amount'] = df_filtrado['amount'].values
                                
                                # Calcular la varianza explicada por cada componente
                                explained_variance = pca.explained_variance_ratio_
                                
                                # Gráfico de dispersión con PCA
                                fig = px.scatter(
                                    pca_df,
                                    x='Componente 1',
                                    y='Componente 2',
                                    color='etiqueta_cluster',
                                    hover_name='id_pos',
                                    hover_data=['amount'],
                                    title=f"Visualización 2D de Segmentos (PCA, {explained_variance[0]*100:.1f}% + {explained_variance[1]*100:.1f}% = {sum(explained_variance)*100:.1f}% de la varianza)"
                                )
                                
                                # Añadir anotación sobre qué representa cada eje
                                fig.update_layout(
                                    xaxis_title=f"Componente Principal 1 ({explained_variance[0]*100:.1f}% de varianza)",
                                    yaxis_title=f"Componente Principal 2 ({explained_variance[1]*100:.1f}% de varianza)"
                                )
                            
                            else:
                                # Visualización directa con las dos características seleccionadas
                                fig = px.scatter(
                                    df_filtrado,
                                    x=features_seleccionadas[0],
                                    y=features_seleccionadas[1],
                                    color='etiqueta_cluster',
                                    hover_name='id_pos',
                                    title="Visualización 2D de Segmentos"
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Si usamos PCA, mostrar información sobre las componentes principales
                            if len(features_seleccionadas) > 2:
                                st.markdown("#### Interpretación de Componentes Principales")
                                
                                # Crear DataFrame con las contribuciones de cada variable a las componentes
                                component_df = pd.DataFrame(
                                    pca.components_.T, 
                                    columns=['Componente 1', 'Componente 2'],
                                    index=features_seleccionadas
                                )
                                
                                # Mostrar contribuciones como tabla
                                st.write("Contribución de cada variable a las componentes principales:")
                                st.dataframe(component_df.style.format("{:.3f}"), use_container_width=True)
                                
                                # Visualización alternativa: gráfico de barras
                                fig_contrib = px.bar(
                                    component_df.reset_index(),
                                    x='index',
                                    y=['Componente 1', 'Componente 2'],
                                    barmode='group',
                                    labels={'index': 'Variables', 'value': 'Contribución', 'variable': 'Componente'},
                                    title="Contribución de variables a las componentes principales"
                                )
                                st.plotly_chart(fig_contrib, use_container_width=True)
                        else:
                            # Si solo hay una característica seleccionada
                            st.info("Selecciona al menos 2 características para visualización en 2D")
                    
                    with col2:
                        # Características promedio por cluster
                        cluster_profile = df_filtrado.groupby('etiqueta_cluster')[features_seleccionadas].mean()
                        
                        # Visualizar perfiles
                        fig = px.imshow(
                            cluster_profile.T,
                            text_auto='.2f',
                            title="Perfil de Segmentos",
                            labels=dict(x="Segmento", y="Métrica", color="Valor")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Añadir tabla con conteo de puntos por cluster
                        st.markdown("#### Distribución de Puntos por Segmento")
                        cluster_counts = df_filtrado['etiqueta_cluster'].value_counts().reset_index()
                        cluster_counts.columns = ['Segmento', 'Número de Puntos']
                        
                        # Calcular porcentajes
                        total = cluster_counts['Número de Puntos'].sum()
                        cluster_counts['Porcentaje'] = (cluster_counts['Número de Puntos'] / total * 100).round(1)
                        
                        # Agregar valores promedio de amount por cluster
                        amount_by_cluster = df_filtrado.groupby('etiqueta_cluster')['amount'].mean().reset_index()
                        amount_by_cluster.columns = ['Segmento', 'Valor Real Promedio']
                        cluster_counts = pd.merge(cluster_counts, amount_by_cluster, on='Segmento')
                        
                        # Agregar valores promedio predichos por cluster 
                        pred_by_cluster = df_filtrado.groupby('etiqueta_cluster')['predicted_subscription_value'].mean().reset_index()
                        pred_by_cluster.columns = ['Segmento', 'Valor Predicho Promedio']
                        cluster_counts = pd.merge(cluster_counts, pred_by_cluster, on='Segmento')
                        
                        # Formatear valores monetarios
                        cluster_counts['Valor Real Promedio'] = cluster_counts['Valor Real Promedio'].round(2).apply(lambda x: f"${x:.2f}")
                        cluster_counts['Valor Predicho Promedio'] = cluster_counts['Valor Predicho Promedio'].round(2)
                        
                        # Mostrar tabla
                        st.dataframe(cluster_counts, use_container_width=True)
                        
                        # Gráfico de radar para comparar clusters
                        if len(features_seleccionadas) >= 3:
                            st.markdown("#### Comparación de Segmentos (Gráfico de Radar)")
                            
                            # Normalizar datos para radar chart
                            radar_df = cluster_profile.copy()
                            for col in radar_df.columns:
                                if radar_df[col].max() > 0:
                                    radar_df[col] = radar_df[col] / radar_df[col].max()
                            
                            # Crear radar chart
                            fig = go.Figure()
                            
                            for idx, segment in enumerate(radar_df.index):
                                fig.add_trace(go.Scatterpolar(
                                    r=radar_df.loc[segment].values,
                                    theta=radar_df.columns,
                                    fill='toself',
                                    name=segment
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                title="Comparación de Segmentos (Normalizado)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error en la segmentación: {str(e)}")
            else:
                st.info("Selecciona al menos una característica para segmentación.")
        else:
            st.warning("Se necesitan al menos 5 puntos de venta para realizar la segmentación.")
    
    # PESTAÑA 3: DATOS
    with tab3:
        st.header("Datos de Puntos de Venta")
        
        # Selector de columnas
        todas_columnas = df_filtrado.columns.tolist()
        columnas_mostrar = st.multiselect(
            "Seleccionar columnas para mostrar",
            todas_columnas,
            default=['id_pos', 'amount', 'predicted_subscription_value', 'amount_category', 'predicted_category']
        )
        
        if columnas_mostrar:
            # Añadir checkbox para ordenar por amount
            orden_por_amount = st.checkbox("Ordenar por Valor Real (amount) descendente", value=True)
            
            # Mostrar datos filtrados
            if orden_por_amount:
                df_display = df_filtrado[columnas_mostrar].sort_values('amount', ascending=False)
            else:
                df_display = df_filtrado[columnas_mostrar]
            
            st.dataframe(df_display, use_container_width=True)
            
            # Opción para descargar
            csv = df_display.to_csv(index=False)
            st.download_button(
                "Descargar datos como CSV",
                data=csv,
                file_name="puntos_venta_segmentados.csv",
                mime="text/csv",
            )
        else:
            st.info("Selecciona al menos una columna para mostrar.")
else:
    st.error("No se pudieron cargar los datos. Verifica que el archivo 'df_Clustering_ICP.csv' esté disponible.")