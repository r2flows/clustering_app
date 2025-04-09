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
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

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
        


        # NUEVO: Cálculo del umbral óptimo de matched_score
        # Definimos clientes de alto valor (top 25% por amount)
        high_value_threshold = df['amount'].quantile(0.75)
        df['is_high_value'] = df['amount'] >= high_value_threshold
        
        if 'lat' in df.columns and 'lng' in df.columns:
            df['latitude'] = df['lat']
            df['longitude'] = df['lng']
        elif 'latitude' not in df.columns or 'longitude' not in df.columns:
            # Si no hay coordenadas disponibles, generamos datos ficticios como último recurso
            st.warning("No se encontraron coordenadas reales en el dataset. Utilizando datos ficticios para la demostración.")
            #np.random.seed(42)
            #df['latitude'] = np.random.normal(19.4326, 0.05, size=len(df))
            #df['longitude'] = np.random.normal(-99.1332, 0.05, size=len(df))

        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

# Función para cargar datos de farmacias como leads potenciales
@st.cache_data
def cargar_leads_potenciales():
    try:
        # Cargar el dataset de farmacias
        df_farmacias = pd.read_csv('farmacias_latam.csv')
        
        # Seleccionar y renombrar columnas relevantes
        df_leads = df_farmacias[['name', 'lat', 'lng', 'score', 'administrative_area_level_1', 
                                 'business_status', 'formatted_phone_number', 'vicinity', 'rating', 
                                 'type_1', 'type_2']].copy()
        df_leads['potencial'] = pd.qcut(
            df_leads['score'].rank(method='first'), 
            q=3, 
            labels=['Bajo', 'Medio', 'Alto']
        ).astype(str)
        # Renombrar columnas para consistencia
        df_leads.rename(columns={
            'lat': 'latitude',
            'lng': 'longitude',
            'name': 'nombre_negocio',
            'administrative_area_level_1': 'region',
            'vicinity': 'direccion',
            'type_1': 'tipo_negocio',
            'formatted_phone_number': 'telefono'
        }, inplace=True)
        
        # Agregar un identificador único para cada lead potencial
        df_leads['id_lead'] = 'LEAD-' + df_leads.index.astype(str)
        
        # Clasificar el score en rangos para facilitar el filtrado
        # Asumimos que el score es un indicador de potencial (mayor = mejor)
        df_leads['potencial'] = pd.qcut(
            df_leads['score'].rank(method='first'), 
            q=3, 
            labels=['Bajo', 'Medio', 'Alto']
        ).astype(str)
        
        # Asignar un valor ficticio amount muy bajo para diferenciar de clientes actuales
        df_leads['amount'] = 0
        
        # Marcar como lead potencial (no cliente actual)
        df_leads['es_cliente'] = False
        df_leads['es_lead'] = True
        
        return df_leads
    except Exception as e:
        st.warning(f"No se pudieron cargar los leads potenciales: {str(e)}")
        # Devolver DataFrame vacío si hay error
        return pd.DataFrame()

# Función para combinar clientes actuales y leads potenciales
def combinar_clientes_y_leads(df_clientes, df_leads, incluir_leads=True):
    if df_leads.empty or not incluir_leads:
        # Si no hay leads o no se quieren incluir, devolver solo clientes
        df_clientes['es_cliente'] = True
        df_clientes['es_lead'] = False
        return df_clientes
    
    # Asegurar columnas comunes para la unión
    columnas_comunes = ['latitude', 'longitude', 'amount', 'es_cliente', 'es_lead']
    for col in columnas_comunes:
        if col not in df_clientes.columns:
            if col in ['es_cliente', 'es_lead']:
                df_clientes[col] = True if col == 'es_cliente' else False
            else:
                df_clientes[col] = None
        if col not in df_leads.columns:
            if col in ['es_cliente', 'es_lead']:
                df_leads[col] = False if col == 'es_cliente' else True
            else:
                df_leads[col] = None
    
    # Combinar los DataFrames
    df_combinado = pd.concat([df_clientes, df_leads], ignore_index=True)
    
    return df_combinado


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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis", "🔍 Segmentación", "📋 Datos", "🎯 Adquisición de Leads"])
    
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
                'active_days_coef_normalized'#, 'dynamic_score'
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

    with tab4:
        st.header("🎯 Estrategia de Adquisición de Leads")
        st.markdown("""
        Esta pestaña ayuda a identificar el perfil de clientes ideales basado principalmente en el **matched_score** 
        y otras variables disponibles durante la adquisición de leads.
        """)
        
        # Dividir en secciones
        st.subheader("1️⃣ Análisis del Matched Score en Clientes Exitosos")
        
        # ---------- IMPLEMENTACIÓN PUNTO 1: ANÁLISIS DE MATCHED_SCORE ----------
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de matched_score por valor
            fig = px.histogram(
                df, 
                x='matched_score_normalized',
                color='amount_category',
                marginal="box",
                title="Distribución de Matched Score por Categoría de Valor"
            )
            fig.update_layout(xaxis_title="Matched Score", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular correlación con amount
            corr_matched = df['matched_score_normalized'].corr(df['amount'])
            st.metric("Correlación Matched Score - Valor Real", f"{corr_matched:.3f}")
            
            if corr_matched > 0.3:
                st.success("Hay una correlación significativa entre el matched score y el valor del cliente.")
            elif corr_matched > 0.1:
                st.info("Hay una correlación débil entre el matched score y el valor del cliente.")
            else:
                st.warning("La correlación entre matched score y valor del cliente es muy baja.")
                
        with col2:
            # Calcular umbrales óptimos
            st.markdown("### Definición de Umbrales Óptimos")
            
            # Crear opciones para el percentil de clientes de alto valor
            percentil_opciones = [90, 80, 75, 70, 60, 50]
            percentil_seleccionado = st.selectbox(
                "Definir clientes de alto valor como el top X percentil", 
                percentil_opciones, 
                index=2  # Default al 75%
            )
            
            # Calcular umbral de alto valor basado en percentil seleccionado
            high_value_threshold = df['amount'].quantile(percentil_seleccionado/100)
            df_temp = df.copy()
            df_temp['is_high_value'] = df_temp['amount'] >= high_value_threshold
            
            # Estadísticas sobre clientes de alto valor
            num_high_value = df_temp['is_high_value'].sum()
            pct_high_value = (num_high_value / len(df_temp)) * 100
            
            st.metric("Umbral de Alto Valor", f"${high_value_threshold:.2f}")
            st.metric("Clientes de Alto Valor", f"{num_high_value} ({pct_high_value:.1f}%)")
            
            # Calcular estadísticas de matched_score para clientes de alto valor
            matched_high = df_temp[df_temp['is_high_value']]['matched_score_normalized']
            matched_low = df_temp[~df_temp['is_high_value']]['matched_score_normalized']
            
            # Mostrar estadísticas
            stats_high = {
                "Promedio": matched_high.mean(),
                "Mediana": matched_high.median(),
                "Min": matched_high.min(),
                "Max": matched_high.max(),
                "Percentil 25": matched_high.quantile(0.25),
                "Percentil 75": matched_high.quantile(0.75)
            }
            
            stats_low = {
                "Promedio": matched_low.mean(),
                "Mediana": matched_low.median(),
                "Min": matched_low.min(),
                "Max": matched_low.max(),
                "Percentil 25": matched_low.quantile(0.25),
                "Percentil 75": matched_low.quantile(0.75)
            }
            
            # Crear DataFrame para comparación
            stats_df = pd.DataFrame({
                "Clientes Alto Valor": stats_high,
                "Otros Clientes": stats_low
            }).T
            
            st.markdown("### Estadísticas de Matched Score por Tipo de Cliente")
            st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)
            
            # Calcular umbral óptimo (podemos usar diferentes estrategias)
            # Para este ejemplo, usaremos el percentil 25 de clientes de alto valor
            optimal_threshold = matched_high.quantile(0.25)
            
            st.markdown("### Umbral Óptimo Sugerido para Matched Score")
            st.metric("Umbral Sugerido", f"{optimal_threshold:.3f}")
            st.info(f"Los leads con matched_score ≥ {optimal_threshold:.3f} tienen mayor probabilidad de convertirse en clientes de alto valor.")
            
            # Calcular la tasa de éxito predicha
            success_rate = (df_temp[df_temp['matched_score_normalized'] >= optimal_threshold]['is_high_value'].mean()) * 100
            st.metric("Tasa de Éxito Predicha", f"{success_rate:.1f}%", 
                      delta=f"{success_rate - pct_high_value:.1f}%" if success_rate > pct_high_value else f"{success_rate - pct_high_value:.1f}%")
        
        # Gráfico de curva ROC improvisada
        st.markdown("### Análisis de diferentes umbrales de matched_score")
        
        # Definir varios umbrales
        thresholds = np.linspace(df['matched_score_normalized'].min(), df['matched_score_normalized'].max(), 20)
        results = []
        
        # Calcular precisión y recall para cada umbral
        for threshold in thresholds:
            # Leads seleccionados con este umbral
            selected = df_temp['matched_score_normalized'] >= threshold
            
            # Número total de seleccionados
            total_selected = selected.sum()
            
            # Número de seleccionados que son de alto valor
            true_positives = (selected & df_temp['is_high_value']).sum()
            
            # Precision: proporción de seleccionados que son de alto valor
            precision = true_positives / total_selected if total_selected > 0 else 0
            
            # Recall: proporción de clientes de alto valor que son seleccionados
            recall = true_positives / num_high_value if num_high_value > 0 else 0
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Eficiencia: proporción de todos los clientes que son seleccionados
            efficiency = total_selected / len(df_temp)
            
            results.append({
                'Threshold': threshold,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Efficiency': efficiency,
                'Selected': total_selected
            })
        
        # Crear DataFrame con los resultados
        threshold_df = pd.DataFrame(results)
        
        # Visualizar
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de precisión y recall
            fig = px.line(
                threshold_df, 
                x='Threshold', 
                y=['Precision', 'Recall', 'F1'],
                title="Métricas de Rendimiento por Umbral de Matched Score"
            )
            fig.update_layout(xaxis_title="Umbral de Matched Score", yaxis_title="Valor", legend_title="Métrica")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Gráfico de cantidad seleccionada vs umbral
            fig = px.line(
                threshold_df, 
                x='Threshold', 
                y='Selected',
                title="Número de Leads Seleccionados por Umbral"
            )
            fig.update_layout(xaxis_title="Umbral de Matched Score", yaxis_title="Leads Seleccionados")
            st.plotly_chart(fig, use_container_width=True)
       # Recomendación de umbral optimizado
        # Encontrar el umbral con el máximo F1 score
        best_threshold_idx = threshold_df['F1'].idxmax()
        best_threshold = threshold_df.loc[best_threshold_idx, 'Threshold']
        best_precision = threshold_df.loc[best_threshold_idx, 'Precision']
        best_recall = threshold_df.loc[best_threshold_idx, 'Recall']
        best_f1 = threshold_df.loc[best_threshold_idx, 'F1']
        best_selected = threshold_df.loc[best_threshold_idx, 'Selected']
        
        st.success(f"""
        ### Umbral Óptimo Recomendado: {best_threshold:.3f}
        
        Con este umbral:
        - Precisión: {best_precision:.1%} (proporción de seleccionados que serán de alto valor)
        - Recall: {best_recall:.1%} (proporción de clientes de alto valor que serán seleccionados)
        - F1 Score: {best_f1:.3f}
        - Número de leads a seleccionar: {best_selected} ({best_selected / len(df_temp):.1%} del total)
        """)
        
        # Selector de umbral personalizado
        custom_threshold = st.slider(
            "Prueba diferentes umbrales de matched_score",
            min_value=float(df['matched_score_normalized'].min()),
            max_value=float(df['matched_score_normalized'].max()),
            value=float(best_threshold),
            step=0.01
        )
        
        # Calcular métricas para el umbral seleccionado
        selected_custom = df_temp['matched_score_normalized'] >= custom_threshold
        total_selected_custom = selected_custom.sum()
        true_positives_custom = (selected_custom & df_temp['is_high_value']).sum()
        precision_custom = true_positives_custom / total_selected_custom if total_selected_custom > 0 else 0
        recall_custom = true_positives_custom / num_high_value if num_high_value > 0 else 0
        f1_custom = 2 * (precision_custom * recall_custom) / (precision_custom + recall_custom) if (precision_custom + recall_custom) > 0 else 0
        
        # Mostrar métricas del umbral personalizado
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precisión", f"{precision_custom:.1%}")
        col2.metric("Recall", f"{recall_custom:.1%}")
        col3.metric("F1 Score", f"{f1_custom:.3f}")
        col4.metric("Leads Seleccionados", f"{total_selected_custom} ({total_selected_custom / len(df_temp):.1%})")
             
        st.subheader("2️⃣ Segmentación Geográfica")
        st.markdown("""
        Visualización de la distribución geográfica de clientes y análisis de zonas de alto valor.
        """)
        
        # Crear un mapa con folium
        if 'latitude' in df_filtrado.columns and 'longitude' in df_filtrado.columns:
            st.markdown("### Mapa de Calor de Clientes")
            
            # Crear un mapa centrado en el promedio de las coordenadas
            centro_lat = df['latitude'].mean()
            centro_lon = df['longitude'].mean()
            
            lat_std = df_filtrado['latitude'].std()
            lon_std = df_filtrado['longitude'].std()
    

            # Crear diferentes opciones de visualización de mapa
            map_options = st.radio(
                "Selecciona el tipo de visualización:",
                ["Todos los clientes", "Solo clientes de alto valor", "Mapa de calor por valor"],
                horizontal=True
            )
            dispersion = max(lat_std, lon_std) * 10
            if dispersion < 0.5:
                zoom_start = 12  # Muy concentrado (ej. una ciudad)
            elif dispersion < 1.5:
                zoom_start = 10  # Área metropolitana
            elif dispersion < 5:
                zoom_start = 8   # Región/Estado
            else:
                zoom_start = 6   # País entero
    

            # Crear el mapa base
            m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom_start)
            
            if map_options == "Todos los clientes":
                # Crear un cluster de marcadores
                mc = MarkerCluster()
                
                # Añadir marcadores para cada punto
                for idx, row in df.iterrows():
                    # Color basado en categoría de amount
                    if row['amount_category'] == 'Alto':
                        color = 'red'
                    elif row['amount_category'] == 'Medio':
                        color = 'orange'
                    else:
                        color = 'blue'
                    
                    # Crear popup con información
                    popup_text = f"""
                    <strong>ID: {row['id_pos']}</strong><br>
                    Valor Real: ${row['amount']:.2f}<br>
                    Matched Score: {row['matched_score_normalized']:.3f}<br>
                    Categoría: {row['amount_category']}
                    """
                    
                    # Añadir marcador al cluster
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color=color, icon='store', prefix='fa')
                    ).add_to(mc)
                
                mc.add_to(m)
                
            elif map_options == "Solo clientes de alto valor":
                # Filtrar solo clientes de alto valor
                high_value_df = df[df['is_high_value']]
                
                # Usar MarkerCluster para los puntos
                mc = MarkerCluster()
                
                # Añadir marcadores para cada punto de alto valor
                for idx, row in high_value_df.iterrows():
                    # Crear popup con información
                    popup_text = f"""
                    <strong>ID: {row['id_pos']}</strong><br>
                    Valor Real: ${row['amount']:.2f}<br>
                    Matched Score: {row['matched_score_normalized']:.3f}
                    """
                    
                    # Añadir marcador al cluster
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color='red', icon='star', prefix='fa')
                    ).add_to(mc)
                
                mc.add_to(m)
                
            else:  # Mapa de calor por valor
                # Preparar datos para el mapa de calor
                heat_data = [[row['latitude'], row['longitude'], row['amount']] for idx, row in df.iterrows()]
                
                # Añadir mapa de calor
                from folium.plugins import HeatMap
                HeatMap(heat_data).add_to(m)
            
            # Mostrar el mapa
            folium_static(m, width=1000, height=600)
            
            st.markdown("### Análisis de Zonas de Concentración")
            
            # Utilizamos KMeans para identificar centros de concentración
            from sklearn.cluster import KMeans
            
            # Seleccionar cuántas zonas identificar
            n_zonas = st.slider("Número de zonas a identificar", 2, 10, 5)
            
            # Preparar datos para clusterización geográfica
            geo_data = df[['latitude', 'longitude']].copy()
            
            # Aplicar KMeans para identificar zonas
            kmeans_geo = KMeans(n_clusters=n_zonas, random_state=42, n_init=10)
            df['geo_cluster'] = kmeans_geo.fit_predict(geo_data)
            
            # Calcular el valor promedio por zona
            zona_stats = df.groupby('geo_cluster').agg({
                'id_pos': 'count',
                'amount': 'mean',
                'matched_score_normalized': 'mean',
                'latitude': 'mean',
                'longitude': 'mean',
                'is_high_value': 'mean'
            }).reset_index()
            
            # Renombrar columnas para claridad
            zona_stats.columns = ['Zona', 'Cantidad de Puntos', 'Valor Promedio', 'Matched Score Promedio', 
                                  'Latitud Central', 'Longitud Central', 'Proporción Alto Valor']
            
            # Formatear valores
            zona_stats['Valor Promedio'] = zona_stats['Valor Promedio'].round(2).apply(lambda x: f"${x:.2f}")
            zona_stats['Matched Score Promedio'] = zona_stats['Matched Score Promedio'].round(3)
            zona_stats['Proporción Alto Valor'] = (zona_stats['Proporción Alto Valor'] * 100).round(1).apply(lambda x: f"{x}%")
            
            # Mostrar estadísticas por zona
            st.dataframe(zona_stats, use_container_width=True)
            
            # Crear un mapa con las zonas identificadas
            m_zonas = folium.Map(location=[centro_lat, centro_lon], zoom_start=10)
            
            # Colores para las diferentes zonas
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
            
            # Crear grupos de marcadores para cada zona
            for i in range(n_zonas):
                # Filtrar datos para esta zona
                zona_df = df[df['geo_cluster'] == i]
                
                # Color para esta zona (ciclo si hay más zonas que colores)
                color = colors[i % len(colors)]
                
                # Crear un grupo de marcadores para esta zona
                fg = folium.FeatureGroup(name=f"Zona {i+1}")
                
                # Añadir un marcador central para la zona
                folium.Marker(
                    location=[zona_stats.loc[i, 'Latitud Central'], zona_stats.loc[i, 'Longitud Central']],
                    popup=f"""
                    <strong>Zona {i+1}</strong><br>
                    Puntos: {zona_stats.loc[i, 'Cantidad de Puntos']}<br>
                    Valor Promedio: {zona_stats.loc[i, 'Valor Promedio']}<br>
                    Matched Score: {zona_stats.loc[i, 'Matched Score Promedio']}<br>
                    % Alto Valor: {zona_stats.loc[i, 'Proporción Alto Valor']}
                    """,
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(fg)
                
                # Dibujar un círculo para marcar la zona
                folium.Circle(
                    radius=1000,  # Radio en metros
                    location=[zona_stats.loc[i, 'Latitud Central'], zona_stats.loc[i, 'Longitud Central']],
                    popup=f"Zona {i+1}",
                    color=color,
                    fill=True,
                ).add_to(fg)
                
                # Añadir el grupo al mapa
                fg.add_to(m_zonas)
            
            # Añadir control de capas
            folium.LayerControl().add_to(m_zonas)
            
            # Mostrar el mapa de zonas
            st.markdown("### Mapa de Zonas Identificadas")
            folium_static(m_zonas, width=1000, height=600)
            
            # Recomendación de zonas prioritarias
            # Ordenar zonas por proporción de alto valor
            zona_stats_sorted = zona_stats.copy()
            zona_stats_sorted['Proporción Alto Valor'] = zona_stats_sorted['Proporción Alto Valor'].str.rstrip('%').astype(float)
            zona_stats_sorted = zona_stats_sorted.sort_values('Proporción Alto Valor', ascending=False)
            
            # Mostrar las 3 mejores zonas
            st.markdown("### Zonas Prioritarias Recomendadas")
            for i in range(min(3, len(zona_stats_sorted))):
                zona = zona_stats_sorted.iloc[i]
                st.info(f"""
                **Zona {int(zona['Zona'])+1}: {zona['Proporción Alto Valor']}% de clientes de alto valor**
                - Ubicación central: Lat {zona['Latitud Central']:.4f}, Lon {zona['Longitud Central']:.4f}
                - Puntos de venta: {int(zona['Cantidad de Puntos'])}
                - Matched Score promedio: {zona['Matched Score Promedio']}
                - Valor promedio: {zona['Valor Promedio']}
                """)
        else:
            st.warning("No hay datos de geolocalización disponibles para la visualización en mapa.")
        
        st.subheader("3️⃣ Estrategia de Adquisición por Proximidad")
        st.markdown("""
            Esta sección ayuda a identificar clientes de alto valor existentes y buscar tanto clientes como 
            **prospectos potenciales** cercanos con características similares.
            """)

# Cargar leads potenciales
        df_leads_potenciales = cargar_leads_potenciales()
        st.write(df_leads_potenciales)
# Información sobre leads cargados
        num_leads = len(df_leads_potenciales)
        if num_leads > 0:
            st.success(f"Se han cargado {num_leads:,} leads potenciales del archivo 'farmacias_latam.csv'")
        else:
            st.warning("No se encontraron leads potenciales. La búsqueda se realizará solo con los puntos de venta existentes.")

# Checkbox para incluir leads potenciales en el análisis
        incluir_leads = st.checkbox("Incluir leads potenciales en el análisis", value=True)

# Opciones de filtro para leads potenciales
        if incluir_leads and not df_leads_potenciales.empty:
            st.markdown("### Filtros para Leads Potenciales")
            col1, col2, col3 = st.columns(3)
    
            with col1:
        # Filtro por potencial (score)
                potencial_opciones = ['Todos'] + sorted(df_leads_potenciales['potencial'].unique().tolist())
                potencial_seleccionado = st.selectbox("Potencial del Lead", potencial_opciones)
    
            with col2:
        # Filtro por región/estado
                region_opciones = ['Todas'] + sorted(df_leads_potenciales['region'].dropna().unique().tolist())
                region_seleccionada = st.selectbox("Región/Estado", region_opciones)
    
            with col3:
        # Filtro por tipo de negocio
                tipo_opciones = ['Todos'] + sorted(df_leads_potenciales['tipo_negocio'].dropna().unique().tolist())
                tipo_seleccionado = st.selectbox("Tipo de Negocio", tipo_opciones)
    
    # Aplicar filtros a leads potenciales
            df_leads_filtrados = df_leads_potenciales.copy()
    
            if potencial_seleccionado != 'Todos':
                df_leads_filtrados = df_leads_filtrados[df_leads_filtrados['potencial'] == potencial_seleccionado]
    
            if region_seleccionada != 'Todas':
                    df_leads_filtrados = df_leads_filtrados[df_leads_filtrados['region'] == region_seleccionada]
    
            if tipo_seleccionado != 'Todos':
                df_leads_filtrados = df_leads_filtrados[df_leads_filtrados['tipo_negocio'] == tipo_seleccionado]
    
    # Actualizar contador de leads después de filtrar
            st.metric("Leads potenciales después de filtros", len(df_leads_filtrados))
        else:
            df_leads_filtrados = pd.DataFrame()

# Seleccionar clientes de alto valor para análisis de proximidad
        if 'latitude' in df.columns and 'longitude' in df.columns:
    # Filtrar clientes de alto valor
            high_value_df = df[df['is_high_value']]
            if len(high_value_df) > 0:

    # Crear selectbox para elegir un cliente de alto valor
                selected_high_value = st.selectbox(
                    "Selecciona un cliente de alto valor como referencia:",
                    high_value_df['id_pos'].tolist(),
                    format_func=lambda x: f"ID: {x} - Valor: ${high_value_df[high_value_df['id_pos']==x]['amount'].values[0]:.2f}"
                )
    
    # Obtener datos del cliente seleccionado
                client_data = high_value_df[high_value_df['id_pos'] == selected_high_value].iloc[0]
    
    # Mostrar información del cliente
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.metric("Valor Real", f"${client_data['amount']:.2f}")
    
                with col2:
                    st.metric("Matched Score", f"{client_data['matched_score_normalized']:.3f}")
    
                with col3:
                    st.metric("Valor Predicho", f"{client_data['predicted_subscription_value']:.2f}")
    
    # Slider para definir radio de búsqueda (en kilómetros)
                search_radius_km = st.slider("Radio de búsqueda (km)", 0.5, 20.0, 5.0, 0.5)
    
    # Combinar clientes actuales y leads potenciales
                if incluir_leads and not df_leads_filtrados.empty:
                    df_combined = combinar_clientes_y_leads(df.copy(), df_leads_filtrados, incluir_leads)
                else:
                    df_combined = df.copy()
                    df_combined['es_cliente'] = True
                    df_combined['es_lead'] = False
    
    # Encontrar puntos dentro del radio
    # Haversine para calcular distancia
                def haversine_distance(lat1, lon1, lat2, lon2):
                    """Calcular distancia Haversine entre dos puntos en latitud/longitud"""
                    from math import radians, sin, cos, sqrt, atan2
        
        # Convertir a radianes
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Fórmula haversine
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distance = 6371 * c  # Radio de la Tierra en km
        
                    return distance
    
    # Calcular distancia para cada punto
                df_combined['distance_km'] = df_combined.apply(
                    lambda row: haversine_distance(
                        client_data['latitude'], client_data['longitude'],
                        row['latitude'], row['longitude']
                    ) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else float('inf'),                        
                    axis=1
                    )
    
    # Filtrar puntos dentro del radio (y que no sean el punto de referencia)
                nearby_points = df_combined[
                    (df_combined['distance_km'] <= search_radius_km) & 
                    (~((df_combined['id_pos'] == client_data['id_pos']) & (df_combined['es_cliente'])))
                ].copy()
    
    # Ordenar por distancia
                nearby_points = nearby_points.sort_values('distance_km')
    

                # Calcular scoring de prioridad para leads cercanos
            if not nearby_points[nearby_points['es_lead']].empty:
    # Separar los leads de los clientes
                leads_cercanos = nearby_points[nearby_points['es_lead']].copy()
    
    # Normalizar distancia (más cercano = mejor)
                max_dist = leads_cercanos['distance_km'].max()
                if max_dist > 0:
                    leads_cercanos['dist_norm'] = 1 - (leads_cercanos['distance_km'] / max_dist)
                else:
                    leads_cercanos['dist_norm'] = 1
    
    # Normalizar score si existe
                if 'score' in leads_cercanos.columns:
                    max_score = leads_cercanos['score'].max()
                    if max_score > 0:
                        leads_cercanos['score_norm'] = leads_cercanos['score'] / max_score
                    else:
                        leads_cercanos['score_norm'] = 0.5  # Valor medio si no hay score
                else:
                    leads_cercanos['score_norm'] = 0.5  # Valor medio si no hay score
    
    # Calcular prioridad combinada (50% proximidad, 50% score)
                leads_cercanos['prioridad'] = (leads_cercanos['dist_norm'] * 0.5) + (leads_cercanos['score_norm'] * 0.5)
    
    # Actualizar los leads en nearby_points con la información de prioridad
    # Primero, eliminar los leads existentes
                nearby_points = nearby_points[~nearby_points['es_lead']]
    # Luego, concatenar con los leads actualizados
                nearby_points = pd.concat([nearby_points, leads_cercanos])
    
    # Ordenar por prioridad para leads y por distancia para clientes
                nearby_points_leads = nearby_points[nearby_points['es_lead']].sort_values('prioridad', ascending=False)
                nearby_points_clients = nearby_points[nearby_points['es_cliente']].sort_values('distance_km')
                nearby_points = pd.concat([nearby_points_clients, nearby_points_leads])


    # Contar clientes y leads cercanos
                nearby_clients = nearby_points[nearby_points['es_cliente']].shape[0]
                nearby_leads = nearby_points[nearby_points['es_lead']].shape[0]
    
    # Mostrar número de puntos encontrados
                col1, col2 = st.columns(2)
                col1.metric("Clientes cercanos", nearby_clients)
                col2.metric("Leads potenciales cercanos", nearby_leads)
    
                if len(nearby_points) > 0:
        # Crear un mapa con los puntos
                    m_nearby = folium.Map(
                        location=[client_data['latitude'], client_data['longitude']],
                        zoom_start=13
                    )
        
        # Añadir cliente de referencia
                    folium.Marker(
                        location=[client_data['latitude'], client_data['longitude']],
                        popup=f"""
                        <strong>REFERENCIA - ID: {client_data['id_pos']}</strong><br>
                        Valor: ${client_data['amount']:.2f}<br>
                        Matched Score: {client_data['matched_score_normalized']:.3f}
                        """,
                        icon=folium.Icon(color='red', icon='star', prefix='fa')
                    ).add_to(m_nearby)
        
        # Añadir círculo de radio de búsqueda
                    folium.Circle(
                        radius=search_radius_km * 1000,  # Convertir a metros
                        location=[client_data['latitude'], client_data['longitude']],
                        popup=f"Radio: {search_radius_km} km",
                        color="red",
                        fill=True,
                        fill_opacity=0.1
                    ).add_to(m_nearby)
        
        # Crear clusters para clientes y leads por separado
                    client_cluster = MarkerCluster(name="Clientes Actuales")
                    lead_cluster = MarkerCluster(name="Leads Potenciales")
        
        # Añadir puntos cercanos
                    for idx, row in nearby_points.iterrows():
            # Determinar si es cliente o lead
                        es_cliente = row['es_cliente']
                        es_lead = row['es_lead']
            
                        if es_cliente:
                # Color basado en si es de alto valor o no
                            color = 'green' if row.get('is_high_value', False) else 'blue'
                
                # Ícono basado en matched score comparado con el cliente de referencia
                            if row.get('matched_score_normalized', 0) >= client_data['matched_score_normalized'] * 0.9:
                                icon = 'check-circle'  # Matched score similar o mejor
                            else:
                                icon = 'circle'  # Matched score inferior
                
                # Crear popup con información
                            popup_text = f"""
                            <strong>CLIENTE - ID: {row['id_pos']}</strong><br>
                            Distancia: {row['distance_km']:.2f} km<br>
                            Valor: ${row['amount']:.2f}<br>
                            Matched Score: {row.get('matched_score_normalized', 'N/A'):.3f}<br>
                            {'<span style="color:green">☑️ Cliente de Alto Valor</span>' if row.get('is_high_value', False) else '<span style="color:blue">Cliente Regular</span>'}
                            """
                
                # Añadir marcador al cluster de clientes
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=folium.Popup(popup_text, max_width=300),
                                icon=folium.Icon(color=color, icon=icon, prefix='fa')
                            ).add_to(client_cluster)
            
                        elif es_lead:
                # Para leads potenciales
                            color = 'purple'
                
                # Determinar icono basado en potencial
                            icon = 'shopping-cart'
                            if 'potencial' in row and row['potencial'] == 'Alto':
                                icon = 'bolt'  # Alto potencial
                
                # Crear popup con información
                            popup_text = f"""
                <strong>LEAD POTENCIAL - ID: {row.get('id_lead', 'N/A')}</strong><br>
                <strong>Nombre: {row.get('nombre_negocio', 'Sin nombre')}</strong><br>
                Distancia: {row['distance_km']:.2f} km<br>
                Dirección: {row.get('direccion', 'N/A')}<br>
                Región: {row.get('region', 'N/A')}<br>
                Tipo: {row.get('tipo_negocio', 'N/A')}<br>
                Score: {row.get('score', 'N/A')}<br>
                Potencial: {row.get('potencial', 'N/A')}<br>
                Teléfono: {row.get('telefono', 'N/A')}
                """
                
                # Añadir marcador al cluster de leads
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=folium.Popup(popup_text, max_width=300),
                                icon=folium.Icon(color=color, icon=icon, prefix='fa')
                            ).add_to(lead_cluster)
        
        # Añadir los clusters al mapa
                    client_cluster.add_to(m_nearby)
                    lead_cluster.add_to(m_nearby)
        
        # Añadir control de capas
                    folium.LayerControl().add_to(m_nearby)
        
        # Mostrar el mapa
                    st.markdown("### Mapa de Puntos Cercanos")
                    folium_static(m_nearby, width=1000, height=600)
        
        # Crear pestañas para mostrar tablas de clientes y leads por separado
                    tab_clients, tab_leads = st.tabs(["📊 Clientes Cercanos", "🎯 Leads Potenciales"])
        
                    with tab_clients:
            # Filtrar solo clientes
                        nearby_clients_df = nearby_points[nearby_points['es_cliente']].copy()
            
                        if not nearby_clients_df.empty:
                            st.markdown("### Detalle de Clientes Cercanos")
                # Seleccionar columnas relevantes para clientes
                            client_cols = [
                            'id_pos', 'distance_km', 'amount', 'matched_score_normalized', 
                            'is_high_value', 'amount_category'
                            ]
                            client_cols = [col for col in client_cols if col in nearby_clients_df.columns]
                
                            nearby_display = nearby_clients_df[client_cols].copy()
                
                # Formatear para mejor visualización
                            nearby_display['distance_km'] = nearby_display['distance_km'].round(2)
                            if 'matched_score_normalized' in nearby_display.columns:
                                nearby_display['matched_score_normalized'] = nearby_display['matched_score_normalized'].round(3)
                            if 'amount' in nearby_display.columns:
                                nearby_display['amount'] = nearby_display['amount'].round(2).apply(lambda x: f"${x:.2f}")
                            if 'is_high_value' in nearby_display.columns:
                                nearby_display['is_high_value'] = nearby_display['is_high_value'].map({True: '✅ Sí', False: '❌ No'})
                
                # Renombrar columnas
                            column_names = {
                                    'id_pos': 'ID POS', 
                                    'distance_km': 'Distancia (km)', 
                                    'amount': 'Valor', 
                                    'matched_score_normalized': 'Matched Score', 
                                    'is_high_value': 'Alto Valor', 
                                    'amount_category': 'Categoría'
                                }
                            nearby_display.rename(columns={k: v for k, v in column_names.items() if k in nearby_display.columns}, inplace=True)
                
                # Mostrar tabla
                            st.dataframe(nearby_display, use_container_width=True)
                
                # Descargar clientes cercanos
                            csv_clients = nearby_display.to_csv(index=False)
                            st.download_button(
                                    "Descargar datos de clientes cercanos",
                                    data=csv_clients,
                                    file_name="clientes_cercanos.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.info("No se encontraron clientes cercanos en el radio especificado.")
        
                    with tab_leads:
            # Filtrar solo leads
                        nearby_leads_df = nearby_points[nearby_points['es_lead']].copy()
            
                        if not nearby_leads_df.empty:
                            st.markdown("### Detalle de Leads Potenciales Cercanos")
                # Seleccionar columnas relevantes para leads
                            lead_cols = [
                                'id_lead', 'nombre_negocio', 'distance_km', 'direccion', 
                                'region', 'tipo_negocio', 'score', 'potencial', 'telefono'
                                ]
                            lead_cols = [col for col in lead_cols if col in nearby_leads_df.columns]
                
                            leads_display = nearby_leads_df[lead_cols].copy()
                
                # Formatear para mejor visualización
                            if 'distance_km' in leads_display.columns:
                                    leads_display['distance_km'] = leads_display['distance_km'].round(2)
                            if 'score' in leads_display.columns:
                                    leads_display['score'] = leads_display['score'].round(2)
                
                # Renombrar columnas
                            column_names = {
                    'id_lead': 'ID Lead', 
                    'nombre_negocio': 'Nombre', 
                    'distance_km': 'Distancia (km)', 
                    'direccion': 'Dirección', 
                    'region': 'Región', 
                    'tipo_negocio': 'Tipo', 
                    'score': 'Score', 
                    'potencial': 'Potencial', 
                    'telefono': 'Teléfono'
                                }
                            leads_display.rename(columns={k: v for k, v in column_names.items() if k in leads_display.columns}, inplace=True)
                
                # Mostrar tabla
                            st.dataframe(leads_display, use_container_width=True)
                
                # Descargar leads potenciales
                            csv_leads = leads_display.to_csv(index=False)
                            st.download_button(
                                    "Descargar datos de leads potenciales",
                                    data=csv_leads,
                                    file_name="leads_potenciales_cercanos.csv",
                                    mime="text/csv",
                                )
                
                # Análisis de potencial en la zona
                            st.markdown("### Análisis de Leads Potenciales en la Zona")
                
                # Crear métricas de potencial
                            if 'potencial' in nearby_leads_df.columns:
                                potencial_counts = nearby_leads_df['potencial'].value_counts()
                                alto_potencial = potencial_counts.get('Alto', 0)
                                medio_potencial = potencial_counts.get('Medio', 0)
                                bajo_potencial = potencial_counts.get('Bajo', 0)
                    
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Leads de Alto Potencial", alto_potencial)
                                col2.metric("Leads de Medio Potencial", medio_potencial)
                                col3.metric("Leads de Bajo Potencial", bajo_potencial)
                
                # Mostrar gráfico de distribución de potencial
                            if len(nearby_leads_df) >= 3 and 'potencial' in nearby_leads_df.columns:
                                fig = px.pie(
                                nearby_leads_df, 
                                names='potencial',
                                title=f"Distribución de Potencial en Leads Cercanos (Radio: {search_radius_km} km)",
                                color='potencial',
                                color_discrete_map={'Alto': '#2ecc71', 'Medio': '#f39c12', 'Bajo': '#e74c3c'}
                            )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No se encontraron leads potenciales en el radio especificado.")
        
        # Análisis de similitud en la zona (considerando solo clientes)
                    nearby_clients_df = nearby_points[nearby_points['es_cliente']].copy()
                    if not nearby_clients_df.empty:
                        st.markdown("### Análisis de Similitud en la Zona")
            
            # Calcular estadísticas de la zona vs cliente de referencia
                        zone_stats = {
                            'matched_score_mean': nearby_clients_df.get('matched_score_normalized', pd.Series()).mean(),
                            'matched_score_median': nearby_clients_df.get('matched_score_normalized', pd.Series()).median(),
                            'amount_mean': nearby_clients_df['amount'].mean(),
                            'amount_median': nearby_clients_df['amount'].median(),
                            'high_value_rate': nearby_clients_df.get('is_high_value', pd.Series()).mean() * 100
                            }
            
            # Mostrar comparación
                        col1, col2 = st.columns(2)
            
                        with col1:
                            st.markdown("#### Cliente de Referencia")
                            st.metric("Matched Score", f"{client_data.get('matched_score_normalized', 'N/A'):.3f}")
                            st.metric("Valor", f"${client_data['amount']:.2f}")

                        with col2:
                            st.markdown("#### Promedio en la Zona")
                            if pd.notna(zone_stats['matched_score_mean']):
                                st.metric("Matched Score Promedio", f"{zone_stats['matched_score_mean']:.3f}", 
                                            delta=f"{zone_stats['matched_score_mean'] - client_data['matched_score_normalized']:.3f}")
                            st.metric("Valor Promedio", f"${zone_stats['amount_mean']:.2f}", 
                                        delta=f"${zone_stats['amount_mean'] - client_data['amount']:.2f}")
                            st.metric("Tasa de Alto Valor", f"{zone_stats['high_value_rate']:.1f}%")
            
            # Conclusión sobre la zona
                        if (zone_stats['matched_score_mean'] >= client_data['matched_score_normalized'] * 0.9 and 
                            zone_stats['high_value_rate'] >= 30):
                            st.success("""
                ### ✅ Zona de Alta Prioridad
                Esta zona tiene características muy similares al cliente de referencia y una alta tasa de clientes de valor.
                Se recomienda dar prioridad a la adquisición de leads en esta área.
                """)
                        elif (zone_stats['matched_score_mean'] >= client_data['matched_score_normalized'] * 0.7 and 
                            zone_stats['high_value_rate'] >= 20):
                            st.info("""
                ### ℹ️ Zona de Prioridad Media
                Esta zona tiene características moderadamente similares al cliente de referencia.
                Puede ser un área de oportunidad para la adquisición de leads.
                """)
                        else:
                            st.warning("""
                ### ⚠️ Zona de Baja Prioridad
                Esta zona tiene características diferentes al cliente de referencia.
                Se recomienda evaluar otras áreas para la adquisición de leads.
                """)
            
            # Sección adicional para análisis de leads potenciales si existen
                        if incluir_leads and not nearby_points[nearby_points['es_lead']].empty:
                            st.markdown("### 🎯 Estrategia de Contacto para Leads Potenciales")
                
                # Crear un score combinado para priorizar leads
                            leads_cercanos = nearby_points[nearby_points['es_lead']].copy()
                
                # Normalizar distancia (más cercano = mejor)
                            max_dist = leads_cercanos['distance_km'].max()
                            if max_dist > 0:
                                leads_cercanos['dist_norm'] = 1 - (leads_cercanos['distance_km'] / max_dist)
                            else:
                                leads_cercanos['dist_norm'] = 1
                
                # Normalizar score si existe
                            if 'score' in leads_cercanos.columns:
                                max_score = leads_cercanos['score'].max()
                                if max_score > 0:
                                    leads_cercanos['score_norm'] = leads_cercanos['score'] / max_score
                                else:
                                    leads_cercanos['score_norm'] = 0
                            else:
                                leads_cercanos['score_norm'] = 0.5  # Valor medio si no hay score
                
                # Calcular prioridad combinada (50% proximidad, 50% score)
                            leads_cercanos['prioridad'] = (leads_cercanos['dist_norm'] * 0.5) + (leads_cercanos['score_norm'] * 0.5)
                
                # Ordenar por prioridad
                            leads_priorizados = leads_cercanos.sort_values('prioridad', ascending=False)
                
                # Mostrar top leads recomendados
                            st.markdown("#### Leads Prioritarios Recomendados")
                
                # Limitar a los 5 mejores leads
                            top_leads = leads_priorizados.head(5)
                
                            for i, (idx, lead) in enumerate(top_leads.iterrows()):
                    # Determinar color basado en prioridad
                                if lead['prioridad'] > 0.8:
                                    color = "success"
                                elif lead['prioridad'] > 0.6:
                                    color = "info"
                                else:
                                    color = "warning"
                    
                    # Mostrar información del lead
                                st.markdown(f"""
                    <div style="border-left: 5px solid {'green' if color=='success' else 'blue' if color=='info' else 'orange'}; padding-left: 10px;">
                    <h5>Lead #{i+1}: {lead.get('nombre_negocio', 'Sin nombre')}</h5>
                    <p>
                    <strong>Prioridad:</strong> {lead['prioridad']:.2f} | 
                    <strong>Distancia:</strong> {lead['distance_km']:.2f} km | 
                    <strong>Score:</strong> {lead.get('score', 'N/A')} | 
                    <strong>Potencial:</strong> {lead.get('potencial', 'N/A')}
                    </p>
                    <p>
                    <strong>Dirección:</strong> {lead.get('direccion', 'N/A')}<br>
                    <strong>Región:</strong> {lead.get('region', 'N/A')}<br>
                    <strong>Tipo:</strong> {lead.get('tipo_negocio', 'N/A')}<br>
                    <strong>Teléfono:</strong> {lead.get('telefono', 'N/A')}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Estrategia de comunicación
                            st.markdown("""
                #### Estrategia de Comunicación Recomendada
                
                Para maximizar la efectividad del contacto con estos leads potenciales:
                
                1. **Enfoque personalizado**: Mencionar la presencia de clientes similares exitosos en la zona.
                
                2. **Beneficios localizados**: Destacar los beneficios específicos que han experimentado otros negocios en la misma área geográfica.
                
                3. **Prueba social**: Utilizar testimonios anónimos de clientes cercanos de alto valor.
                
                4. **Oferta geolocalizada**: Crear una oferta especial para negocios en esta zona específica.
                
                5. **Seguimiento escalonado**: Comenzar con los leads de mayor prioridad y continuar con los siguientes según los resultados obtenidos.
                """)
                
                # Opción para descargar la lista de leads priorizados
                            csv_priorizados = leads_priorizados.to_csv(index=False)
                            st.download_button(
                            "Descargar lista completa de leads priorizados",
                            data=csv_priorizados,
                            file_name="leads_priorizados.csv",
                            mime="text/csv",
                            )
                else:
                    st.warning(f"No se encontraron puntos dentro del radio de {search_radius_km} km.")
        
        # Sugerir aumentar el radio
                    st.info("Prueba aumentar el radio de búsqueda para encontrar más puntos.")
        else:
            st.warning("No se encontraron clientes de alto valor para usar como referencia. Intenta ajustar los filtros.")
else:
    st.warning("No hay datos de geolocalización disponibles para el análisis de proximidad.")