import streamlit as st #ui
import pypdf #to read and extract from the pdf
import csv,re,os #csv to store chat history, re to display a clean output, os for path
import google.genai as ai #access to the ai model - functions to generate and embed text
import numpy as np #for cosine similarity
from operator import itemgetter as ig #sorting
import pandas as pd

#CHUNK is a string
#CHUNKS is a list
st.title("PALANTIR")
api=ai.Client(api_key=st.secrets["x"])
chunks=[] #LIST
q=""
#the user can either select a suggestion to put as question, or input their own question

def cosine(Qv,Av):  #function to calculate cosine similarity
        #cosine similarity=dot product of the two vectors/ cross product of magnitude of the two vectors
    dp=np.dot(Qv, Av) #calculates dot product for Qv and Av
    Qlen=np.linalg.norm(Qv) #magnitude of Qv
    Alen=np.linalg.norm(Av) #magnitude of Av
    cp=Qlen*Alen #getting the cross product
    cos=dp/cp #finding cosine similarity with this formula :: -1 (opp), +1 (same)
    return cos

if "suggestions" not in st.session_state:
    st.session_state.suggestions=[] #session_state stores the suggestions like key-value

csvh="history.csv"
  
if not os.path.exists(csvh): #creates csv file in the form of speaker,text if it isn't alr available
    df=pd.DataFrame(columns=['Speaker', 'Text'])
    df.to_csv(csvh, index=False)

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
               
    if chunks:
        q_us="summarize the document"
        Qo_us=api.models.embed_content(model="embedding-001",contents=[q_us]) #converts the summary of the document to vector
        Qv_us=Qo_us.embeddings[0].values #returns the vector objects
        
        tc_us=[] #'tuple of c (cos and chunk) for user's selected suggestion' stores tuples of form (cos similarity result, chunk)            
        for chunk in chunks:
            Ao1=api.models.embed_content(model='embedding-001', contents=[chunk]) #answer objects
            Av1=Ao1.embeddings[0].values #vector for chunk
            
            cos1=cosine(Qv_us,Av1)
            tc_us.append((cos1,chunk))
        tc_us.sort(key=ig(0), reverse=True) #sorting the tuple by index 0 - cosine similarity - in descending order
            
        top_us=[]
        i=0
        while i<3 and i<len(tc_us): #stops when chunk is third most similar
            top_us.append(tc_us[i][1]) #tc_us[i][0] is the cosine value, tc_us[i][1] is the chunk, adds the chunk to the list - top
            i+=1
                
        top3_us=""
        for chunk in top_us:
            top3_us+=chunk+"\n" 
            
        try:
            Qsug=f'''
            based on {top3_us} suggest 3 good, specific, professional, highly concise questions clearly in only 1 line'''
            sug1=api.models.generate_content(model='gemini-2.5-pro',contents=[Qsug])
            sug2=re.sub(r'<[^>]+>*','',sug1.text)
            suggestions=re.findall(r'\d+\.\s*(.+)', sug2)
            st.session_state["suggestions"]=suggestions
        except:
            suggestions=[]
        
        if suggestions:
            st.markdown("*Palantir's Suggestions:*")
            for s in st.session_state["suggestions"]:
                st.markdown(f"🔮 *{s}*\n")
            st.info("You can ask your own questions too", icon=":material/info:") #icon sets the icon in the format ":material/<name of the symbol>:"
        else:
            st.warning("Couldn't generate suggestions. Please try uploading the file again or ask your own question.", icon=":material/warning:")
        
        quest=st.text_input("What is your question?", key="uq")

        if quest.strip():
            q = quest.strip()
        else:
            q = ""
                
        if q:
            Qo_uq=api.models.embed_content(model='embedding-001',contents=[q]) #the question objects
            Qv_uq=Qo_uq.embeddings[0].values #vector for question
            
            tc_uq=[] #'tuple of c (cos and chunk) for user's question' stores tuples of form (cos similarity result, chunk) 
            for chunk in chunks:
                Ao2=api.models.embed_content(model='embedding-001', contents=[chunk]) #answer objects
                Av2=Ao2.embeddings[0].values #vector for chunk
    
                cos2=cosine(Qv_uq,Av2)
                tc_uq.append((cos2,chunk))
            tc_uq.sort(key=ig(0), reverse=True) #sorting the tuple by index 0 - cosine similarity - in descending order
            
            top_uq=[]
            i=0
            while i<3 and i<len(tc_uq): #stops when chunk is third most similar
                top_uq.append(tc_uq[i][1]) #tc_uq[i][0] is the cosine value, tc_uq[i][1] is the chunk, adds the chunk to the list - top_uq
                i+=1
                    
            top3_uq=""
            for chunk in top_uq:
                top3_uq+=chunk+"\n" 
            
            if top3_uq:
                top=top3_uq
            else:
                top=top3_us
                
            prompt=f'''
            based on {top}, answer the following question accurately and concisely: {q}.
            Please speak in a professional tone. Greet the User first, followed by this emoji: "🔮".
            answer as if you are an expert data analyst and do not share irrelevant details.
            keep your thinking process to yourself.'''
            
            response1=api.models.generate_content(model='gemini-2.5-pro', contents=[prompt])
            response2=re.sub(r'<[^>]+>', '', response1.text) #removes unnecessary responses
            st.session_state['response']=response2
            
            user=pd.DataFrame([{"Speaker": "**You**", "Text": q}])
            user.to_csv(csvh, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL) #appends, no header or index
            bot=pd.DataFrame([{"Speaker": "**AI Data Analyst**", "Text": response2}])
            bot.to_csv(csvh, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL) #appends, no header or index
            
            if 'response' in st.session_state:
                st.write(st.session_state['response'])
                
                if os.path.exists(csvh) and os.path.getsize(csvh)>0:
                    with st.expander("View chat history"):
                        hist=pd.read_csv(csvh)
                        if {'Speaker', 'Text'}.issubset(hist.columns):
                            for index, row in hist.iterrows():
                                speaker=row['Speaker']
                                text=row['Text']
                                if pd.notnull(text):
                                    st.markdown(f"{speaker}:")
                                    st.markdown(f"{text.strip()}")
                                else:
                                    st.markdown(f"AI Data Analyst: {row['Text']}")

                        if st.button("Clear history"):
                            ch=pd.DataFrame(columns=['Speaker', 'Text'])
                            ch.to_csv(csvh, index=False)
                            st.success("Chat history cleared. Please reload.")
                else:
                    st.info("No chat history yet.")

                    