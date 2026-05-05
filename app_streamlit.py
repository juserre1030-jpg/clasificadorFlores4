import streamlit as st
import joblib
import pandas as pd

# Título de la aplicación
st.title('Clasificador de especies de flores de Iris')
st.write('Ingrese las características de la flor para predecir su especie.')

# Cargar el modelo
# Asegúrate de que el archivo del modelo esté accesible (ej. en el mismo directorio que este script)
try:
    model_streamlit = joblib.load('/content/neural_network_model (1).joblib')
    st.success('Modelo cargado exitosamente.')
except FileNotFoundError:
    st.error("Error: El archivo 'neural_network_model (1).joblib' no se encontró. Asegúrate de que está en la ruta correcta.")
    st.stop()

# Entradas del usuario
sepal_length = st.number_input('Largo del sépalo (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0, value=0.3, step=0.1)
petal_length = st.number_input('Largo del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Botón para hacer la predicción
if st.button('Predecir especie'):
    # Preparar los datos de entrada en un DataFrame
    flower_features_streamlit = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    # Realizar la predicción
    prediction_streamlit = model_streamlit.predict(flower_features_streamlit)

    # Mostrar la predicción
    st.subheader('Resultado de la predicción:')
    st.success(f'La especie predicha para la flor es: **{prediction_streamlit[0]}**')
