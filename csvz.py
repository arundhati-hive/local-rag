import streamlit as st
import ollama
import pandas as pd
import re

f=st.file_uploader("upload a csv file", type='csv')
chunks=[]
score=0
content=""

if f:
    df=pd.read_csv(f)
    st.write("preview")
    st.dataframe(df.head())
    for row in df.values:
        lst=[]
        for cell in row:
            lst.append(str(cell))
        line= " ".join(lst)
        content+=line + "\n"

    for i in range(0, len(content), 300): #step count 300 because chunk size is 300
        chunks.append(content[i:i + 300])
    q=st.text_input("ask a question")
    if q:
        Qemb=ollama.embeddings(model='nomic-embed-text', prompt=q)['embedding']
        Cbest=""
        Sbest=-1
        for chunk in chunks:
            Cemb=ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
            for i in range (0, len(Qemb)):
                score+= Qemb[i]*Cemb[i]
            if score>Sbest:
                Sbest=score
                Cbest=chunk
                
        prompt=f'''based on {Cbest}, answer the following question accurately and concisely: {q}.
Please speak in a professional tone.
answer as if you are an expert data analyst and do not share irrelevant details.keep your thinking process to yourself.
'''
        ans=ollama.generate('phi3', prompt)
        response=re.sub(r'<[^>]+>', '', ans['response'])
        st.write(response)
