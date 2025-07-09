import streamlit as st 
import pypdf
import ollama
import re
import pandas as pd

f=st.file_uploader("upload your file",type='pdf')
chunks=[]

if f:
    read=pypdf.PdfReader(f)
    content=""
    pageslst=read.pages
    
    for page in pageslst:
        txt=page.extract_text()
        if txt:
            content+=txt
            
    for i in range(0,len(content),300): # iterated the entire content and splits by the chunk size 300
        chunk=content[i:i+300] #chunk is a dictionary with the start and end location of the chunks
        chunks.append(chunk) #all chunks are added to list - chunks
        
    q=st.text_input("enter your question")
    
    if q:
        Qemb=ollama.embeddings(model='nomic-embed-text', prompt=q)['embedding'] #embeds question
        Cbest="" #placeholder for where the best chunk will come
        Sbest=-1 #the higher the score, the better the chunk is
        
        for chunk in chunks:
            Cemb=ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding'] #embeds chunk
            score=0
            for i in range(0, len(Qemb)):
                score+=Qemb[i]*Cemb[i] #multiplying to see which score will be the highest
            if score>Sbest:
                Sbest=score
                Cbest=chunk
        prompt=f'''
        based on {Cbest}, answer the following question accurately and concisely: {q}.
        Please speak in a professional tone. Greet the User first.
        answer as if you are an expert data analyst and do not share irrelevant details.
        keep your thinking process to yourself.'''
        ans=ollama.generate('phi3',prompt)
        response=re.sub(r'<[^>]+>', '', ans['response'])
        st.write(response)