import base64
import streamlit as st


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


def header(url, size="36", color="D52059", font="'Helvetica', 'Arial', sans-serif"):
    st.markdown(f'<p style="background-color:#FFFFFF;color:#{color};font-size:{size}px;font-family:{font};border-radius:20%;">{url}</p>',
                unsafe_allow_html=True)

def text_w_background(url, size="24",
                      font="'Helvetica', 'Arial', sans-serif",
                      background="FFFFFF"):
    st.markdown(f'<p style="background-color:#{background};font-size:{size}px;font-family:{font};border-radius:20%;">{url}</p>',
                unsafe_allow_html=True)
