import streamlit as st #ui
import pypdf #to read and extract from the pdf
import re #to display a clean output
import google.genai as ai #access to the ai model - functions to generate and embed text
import numpy #for cosine similarity
from operator import itemgetter as ig #sorting

#CHUNK is a string
#CHUNKS is a list
st.title("PALANTIR")
api=ai.Client(api_key=st.secrets["x"])
chunks=[] #LIST
q=""

f=st.file_uploader("Upload your file",type='pdf')

if f:
    read=pypdf.PdfReader(f)
    content="" #stores the entire text
    pageslst=read.pages #list of all pages in the pdf
    
    for page in pageslst:
        txt=page.extract_text() #pulls plain text from pdf
        if txt:
            content+=txt
            
    for i in range(0,len(content),2000): # iterates the entire content and splits by the chunk size 2000
        chunk=content[i:i+2000] #chunk is a string of 2000 characters
        chunks.append(chunk) # chunks here is a list of chunk strings.
        
        Qo=api.models.embed_content(model='embedding-001',contents=[q]) #the question objects
        Qv=Qo.embeddings[0].values #vector for question
        
        tc=[] #'tuple of c (cos and chunk)' stores tuples of form (cos similarity result, chunk)
        
        for chunk in chunks:
            Ao=api.models.embed_content(model='embedding-001', contents=[chunk]) #answer objects
            Av=Ao.embeddings[0].values #vector for chunk
            
            #cosine similarity = dot product of the two vectors/ cross product of magnitude of the two vectors
            dp=numpy.dot(Qv, Av) #calculates dot product for Qv and Av
            Qlen=numpy.linalg.norm(Qv) #magnitude of Qv
            Alen=numpy.linalg.norm(Av) #magnitude of Av
            cp=Qlen*Alen #getting the cross product
            cos=dp/cp #finding cosine similarity with this formula :: -1 (opp), +1 (same)
        
            tc.append((cos,chunk))
        tc.sort(key=ig(0), reverse=True) #sorting the tuple by index 0 - cosine similarity - in descending order
            
        top=[]
        i=0
        while i<3 and i<len(tc): #stops when chunk is third most similar
            top.append(tc[i][1]) #tc[i][0] is the cosine value, tc[i][1] is the chunk, adds the chunk to the list - top
            i+=1
                
        top3=""
        for chunk in top:
            top3+=chunk+"\n" 
            
        Qsug=f'''based on {top3}, suggest 3 good, specific, professional concise questions clearly'''
        sug1=api.models.generate_content(model='gemini-2.5-pro',contents=[Qsug])
        sug2=re.sub(r'<[^>]+>*', '', sug1.text)
        suggestions=re.findall(r'\d+\.\s*(.+)', sug2)
        Qops=["Choose one of the following:"]+suggestions
        
        selected=st.selectbox("Palantir's top 3 suggestions:", Qops)      
        quest=st.text_input("Ask your question")
        
        if quest:
            q=quest
        elif selected!=Qops[0]:
            q=selected
        else:
            q=""
        
            
        prompt=f'''
        based on {top3}, answer the following question accurately and concisely: {q}.
        Please speak in a professional tone. Greet the User first, followed by this emoji: "🔮".
        answer as if you are an expert data analyst and do not share irrelevant details.
        keep your thinking process to yourself.'''
        
        response1=api.models.generate_content(model='gemini-2.5-pro', contents=[prompt])
        response2=re.sub(r'<[^>]+>', '', response1.text) #removes unnecessary responses
        st.write(response2)