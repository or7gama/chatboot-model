import streamlit as st
from chatbot import predict_class, get_response, intents

st.set_page_config(page_title='Chat Academic',page_icon='🤖',layout='centered')

def display_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_new_message(prompt, messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    # Implementación del algoritmo de la IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)

    with st.chat_message("assistant"):
        st.markdown(res)
    messages.append({"role": "assistant", "content": res})

# Configuración de la interfaz
st.title('Chatbot Academic 🎓')

# Inicialización de estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar mensajes existentes
display_messages(st.session_state.messages)

# Manejo del primer mensaje de bienvenida
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("👋🏻 Hola, ¿Cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "👋🏻 Hola, ¿Cómo puedo ayudarte?"})
    st.session_state.first_message = False

# Captura de entrada del usuario
prompt = st.chat_input("Escribe tus comentarios aquí:")
if prompt:
    handle_new_message(prompt, st.session_state.messages)

