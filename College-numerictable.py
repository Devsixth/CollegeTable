import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import plotly.express as px
from PIL import Image


st.title('Course Review')

df=pd.read_csv("college-1.1.csv")



sak=df[df['College']=="SAK"]
sat=df[df['College']=="SAT"]
rgt=df[df['College']=="RGT"]
srs=df[df['College']=="SRS"]
nkt=df[df['College']=="NKT"]
sht=df[df['College']=="SHT"]
ssk=df[df['College']=="SSK"]
msp=df[df['College']=="MSP"]
apt=df[df['College']=="APT"]



st.sidebar.title('College')
side = st.sidebar.radio('Select the college:', ["All","SAK", "SAT", "SRS", "NKT", "RGT", "SHT", "SSK", "MSP","APT"], key="options")



tab1, tab2, tab3, tab4, tab5  = st.tabs(['Instructor Specific', 'Course Specific', 'Student Specific', 'Overall Evaluation', 'Overall Score'])

with tab1:
    st.header("Instructor Specific")
    options2=st.selectbox('Select the Question', ["The instructor communicated clearly and was easy to understand", "The instructor encouraged questions from students", "The standards of performance for the course were clearly communicated", "The instructor systematically relates course concepts, summarizes main points, and emphasizes material", "The instructor showed an interest in helping students learn"], key="options2")
    
    
#ALL--    
    if options2 == "The instructor communicated clearly and was easy to understand" and side == "All":
        
       
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
           
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#SAK---        
    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "SAK":
        
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#---SAT       

    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#-----srs        
    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "SRS":


#plot        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
        
#---nkt--
        
    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "NKT":
        

        
#plot        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
        
#---rgt----        
        
    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "RGT":
        
        
        
#plot    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')    
        
        df_g = rgt.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
#---sht-----

    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "SHT":
        
       
        
#plot        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
    
        df_g = sht.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#---ssk ----

    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "SSK":
        
        

#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
    
        df_g = ssk.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#msp        
    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "MSP":
      
#plot        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#apt

    elif options2 == "The instructor communicated clearly and was easy to understand" and side == "APT":
        
        
#plot   
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = apt.groupby(['Ins_Communication', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Ins_Communication', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Communication', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
#---------ins_encouragement------------

#all
    if options2 == "The instructor encouraged questions from students" and side == "All":
        
        
#plot        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sak        
    elif options2 == "The instructor encouraged questions from students" and side == "SAK":
        
        
#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sat        
    elif options2 == "The instructor encouraged questions from students" and side == "SAT":
        
#plot    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#srs        
    elif options2 == "The instructor encouraged questions from students" and side == "SRS":
    
#plot    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
#nkt        
    elif options2 == "The instructor encouraged questions from students" and side == "NKT":
    
    
#plot    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#rgt        
    elif options2 == "The instructor encouraged questions from students" and side == "RGT":
     
        
#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = rgt.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sht        
    elif options2 == "The instructor encouraged questions from students" and side == "SHT":
       
        
#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sht.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#ssk        
    elif options2 == "The instructor encouraged questions from students" and side == "SSK":
        
     
#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = ssk.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#msp        
    elif options2 == "The instructor encouraged questions from students" and side == "MSP":

#plot
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = msp.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#apt        
    elif options2 == "The instructor encouraged questions from students" and side == "APT":
    
#plot   
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = apt.groupby(['Ins_Encouragement', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Ins_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Encouragement', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#------------Course_performance---------------
#all
    if options2 == "The standards of performance for the course were clearly communicated" and side == "All":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
#sak        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "SAK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sat        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#srs        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "SRS":
            
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#nkt        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "NKT":
    
            
       
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#rgt        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "RGT":
              
        

        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = rgt.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sht        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "SHT":
              
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

        
        df_g = sht.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#ssk        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "SSK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g =ssk.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#msp        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "MSP":

       
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#apt        
    elif options2 == "The standards of performance for the course were clearly communicated" and side == "APT":
             
        
    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = apt.groupby(['Course_performance', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Course_performance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_performance', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
#--------------Ins_Course_concept

    #all
    if options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "All":
        
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sak        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "SAK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sat        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "SAT":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#srs        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "SRS":
            
        
    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#nkt        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "NKT":
    
            
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#rgt        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "RGT":
              
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

        
        df_g = rgt.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#sht        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "SHT":
              
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sht.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#ssk        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "SSK":
        
              
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied') 

        
        df_g = ssk.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#msp        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "MSP":

        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#apt        
    elif options2 == "The instructor systematically relates course concepts, summarizes main points, and emphasizes material" and side == "APT":
             
        
    
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = apt.groupby(['Ins_Course_concept', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Ins_Course_concept', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Ins_Course_concept', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
        #----------Helping_Student-----
        
    if options2 == "The instructor showed an interest in helping students learn" and side == "All":
        
        df_g = df.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options2 == "The instructor showed an interest in helping students learn" and side == "SAK":
        
        df_g = sak.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options2 == "The instructor showed an interest in helping students learn" and side == "SAT":
        
        
        df_g = sat.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options2 == "The instructor showed an interest in helping students learn" and side == "SRS":
        
        df_g = srs.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options2 == "The instructor showed an interest in helping students learn" and side == "NKT":
        
        df_g = nkt.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options2 == "The instructor showed an interest in helping students learn" and side == "RGT":
        
        df_g = rgt.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options2 == "The instructor showed an interest in helping students learn" and side == "SHT":
        
        df_g = sht.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
    
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options2 == "The instructor showed an interest in helping students learn" and side == "SSK":
        
        
        df_g = ssk.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options2 == "The instructor showed an interest in helping students learn"and side == "MSP":
        
        
        df_g = msp.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options2 == "The instructor showed an interest in helping students learn" and side == "APT":
        
        df_g = apt.groupby(['Helping_Student', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Helping_Student', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Helping_Student', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
#------------------- tab2 begin------------    
with tab2:
    st.header("Course Specific")
    options3=st.selectbox('Select the Question', ["I believe the topics covered in this course are important", "This course provides a good balance of theory and practice"
, "The course provides learning materials that improves my understanding and increased my knowledge"
, "The course assignments  are reflective of the course content and syllabus"], key="options3")
    
 #----------- Imp_Topics----------------------
    if options3 == "I believe the topics covered in this course are important" and side == "All":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options3 == "I believe the topics covered in this course are important" and side == "SAK":
        
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options3 == "I believe the topics covered in this course are important" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options3 == "I believe the topics covered in this course are important" and side == "SRS":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
          
        
        df_g = srs.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
    elif options3 == "I believe the topics covered in this course are important" and side == "NKT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
            )
        
    elif options3 == "I believe the topics covered in this course are important" and side == "RGT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = rgt.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "I believe the topics covered in this course are important" and side == "SHT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = sht.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "I believe the topics covered in this course are important" and side == "SSK":
        
        
          
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = ssk.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "I believe the topics covered in this course are important"and side == "MSP":
        
           
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "I believe the topics covered in this course are important" and side == "APT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = apt.groupby(['Imp_Topics', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Imp_Topics', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Imp_Topics', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
#---------- Theory_Practise---------

    if options3 == "This course provides a good balance of theory and practice" and side == "All":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "This course provides a good balance of theory and practice" and side == "SAK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "This course provides a good balance of theory and practice" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "This course provides a good balance of theory and practice" and side == "SRS":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "This course provides a good balance of theory and practice" and side == "NKT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "This course provides a good balance of theory and practice" and side == "RGT":
        
         
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = rgt.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "This course provides a good balance of theory and practice" and side == "SHT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sht.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "This course provides a good balance of theory and practice" and side == "SSK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = ssk.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "This course provides a good balance of theory and practice"and side == "MSP":
        
           
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "This course provides a good balance of theory and practice" and side == "APT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = apt.groupby(['Theory_Practise', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Theory_Practise', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Theory_Practise', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    #------------Learning_Material---------
        
    if options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "All":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "SAK":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sak.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "SRS":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "NKT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "RGT":
        
         
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = rgt.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "SHT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sht.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "SSK":
        
        
          
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = ssk.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge"and side == "MSP":
        
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = msp.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course provides learning materials that improves my understanding and increased my knowledge" and side == "APT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
       
        df_g = apt.groupby(['Learning_Material', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Learning_Material', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Learning_Material', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    #---------------Course_Assignment----------
        
    if options3 == "The course assignments  are reflective of the course content and syllabus" and side == "All":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = df.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "SAK":
        
        
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g =sak.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "SAT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = sat.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "SRS":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = srs.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "NKT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        df_g = nkt.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "RGT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = rgt.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "SHT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = sht.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "SSK":
        
        
          
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = ssk.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "MSP":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
       
        
        df_g = msp.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options3 == "The course assignments  are reflective of the course content and syllabus" and side == "APT":
        
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
        
        
        df_g = apt.groupby(['Course_Assignment', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Course_Assignment', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Course_Assignment', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
#----------------------perception/motivation----------------    
    
with tab3: 
    
    st.header("Perception About the course") 
        
    options5=st.selectbox('Select the Question', ["This course helps me improve my ability to interact with diverse groups of people", "The course challenges me to think deeply about the subject matter", "I attend classes regularly", "The course improves my communication and presentation skills"], key="options5")
    
    if options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="All":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

        df_g = df.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="SAK":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
           
        df_g = sak.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="SAT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sat.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="SRS":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = srs.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )     
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="NKT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = nkt.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )    
        
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="RGT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = rgt.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="SHT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sht.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
        
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="SSK":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = ssk.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="MSP":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = msp.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    elif options5 == "This course helps me improve my ability to interact with diverse groups of people" and side =="APT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = apt.groupby(['Intract_people', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Intract_people', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Intract_people', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )  
#2


    if options5 == "The course challenges me to think deeply about the subject matter" and side =="All":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = df.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )  
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="SAK":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sak.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course challenges me to think deeply about the subject matter" and  side =="SAT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sat.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="SRS":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = srs.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="NKT":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
           
        df_g = nkt.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )     
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="RGT":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = rgt.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )    
        
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="SHT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sht.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="SSK":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = ssk.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )    
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="MSP":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = msp.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
    elif options5 == "The course challenges me to think deeply about the subject matter" and side =="APT":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = apt.groupby(['Subject_Matters', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Subject_Matters', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Subject_Matters', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    #3
      
    if options5 == "I attend classes regularly" and side =="All":
        
            
            
        df_g = df.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
               
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "I attend classes regularly" and side =="SAK":
        
            
            
        df_g = sak.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "I attend classes regularly" and side =="SAT":
            
            
            
        df_g = sat.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "I attend classes regularly" and side =="SRS":
        
            
            
        df_g = srs.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )    
    elif options5 == "I attend classes regularly" and side =="NKT":
        
            
            
        df_g = nkt.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
    elif options5 == "I attend classes regularly" and side =="RGT":
            
        df_g = rgt.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )      
    elif options5 == "I attend classes regularly" and side =="SHT":
            
        df_g = sht.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )       
    elif options5 =="I attend classes regularly" and side =="SSK":
            
        df_g = ssk.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "I attend classes regularly" and side =="MSP":
            
        df_g = msp.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
        
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "I attend classes regularly" and side =="APT":
           
        df_g = apt.groupby(['Regular_Attendance', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Regular_Attendance', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Regular_Attendance', 'Branch', 'Counts', 'Percentage']
            
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
    #4

    if options5 == "The course improves my communication and presentation skills" and side =="All":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = df.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = df.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
        
    elif options5 == "The course improves my communication and presentation skills" and side =="SAK":
        
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sak.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = sak.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "The course improves my communication and presentation skills" and side =="SAT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sat.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = sat.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "The course improves my communication and presentation skills" and side =="SRS":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = srs.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = srs.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    
    elif options5 == "The course improves my communication and presentation skills" and side =="NKT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = nkt.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = nkt.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course improves my communication and presentation skills" and side =="RGT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = rgt.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = rgt.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course improves my communication and presentation skills" and side =="SHT":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = sht.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = sht.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )
        
    elif options5 == "The course improves my communication and presentation skills" and side =="SSK":
            
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = ssk.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = ssk.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "The course improves my communication and presentation skills" and side =="MSP":
           
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = msp.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = msp.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               ) 
        
    elif options5 == "The course improves my communication and presentation skills" and side =="APT":
        st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')
            
        df_g = apt.groupby(['Communication_Skill', 'Branch']).size().reset_index()
        df_g['percentage'] = apt.groupby(['Communication_Skill', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
        df_g.columns = ['Communication_Skill', 'Branch', 'Counts', 'Percentage']
            
        st.table(df_g)
        
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_g)

        st.download_button(
                label="Download",
                data=csv,
                file_name='df_g.csv',
                mime='text/csv',
               )     
#motivation

with tab3: 
    
    st.header("Motivation") 
#1    
    options9=st.selectbox('Select the Question', ["What Influenced your Decision to take this course?", "How did you come to know about this course", "College Encouraged me to take up this course"],key='options9')
    
    if options9 == "What Influenced your Decision to take this course?" and side =="All":
            
            df_g = df.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
    elif options9 == "What Influenced your Decision to take this course?" and side =="SAK":
            
            df_g = sak.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="SAT":
            
            df_g = sat.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="SRS":
            
            df_g = srs.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="NKT":
            
            df_g = nkt.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="RGT":
            
            df_g = rgt.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="SHT":
            
            df_g = sht.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="SSK":
            
            df_g = ssk.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
    elif options9 == "What Influenced your Decision to take this course?" and side =="MSP":
            
            df_g = msp.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "What Influenced your Decision to take this course?" and side =="APT":
            
            df_g = apt.groupby(['Influence', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Influence', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Influence', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
#2

    if options9 == "How did you come to know about this course" and side =="All":
            
            df_g = df.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "How did you come to know about this course" and side =="SAK":
            
            df_g = sak.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "How did you come to know about this course" and side =="SAT":
            
            df_g = sat.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "How did you come to know about this course" and side =="SRS":
            
            df_g = srs.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options9 == "How did you come to know about this course" and side =="NKT":
            
            df_g = nkt.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "How did you come to know about this course" and side =="RGT":
            
            df_g = rgt.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "How did you come to know about this course" and side =="SHT":
            
            df_g = sht.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "How did you come to know about this course" and side =="SSK":
            
            df_g = ssk.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "How did you come to know about this course" and side =="MSP":
            
            df_g = msp.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "How did you come to know about this course" and side =="APT":
            
            df_g = apt.groupby(['Course_Info', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Course_Info', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Info', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    #3

    if options9 == "College Encouraged me to take up this course" and side =="All":
            
            df_g = df.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options9 == "College Encouraged me to take up this course" and side =="SAK":
            
            df_g = sak.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "College Encouraged me to take up this course" and side =="SAT":
            
            df_g = sat.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
                
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            
            
            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options9 == "College Encouraged me to take up this course" and side =="SRS":
            
            df_g = srs.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "College Encouraged me to take up this course" and side =="NKT":
            
            df_g = nkt.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "College Encouraged me to take up this course" and side =="RGT":
            
            df_g = rgt.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
           
            
    elif options9 == "College Encouraged me to take up this course" and side =="SHT":
            
            df_g = sht.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options9 == "College Encouraged me to take up this course" and side =="SSK":
            
            df_g = ssk.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "College Encouraged me to take up this course" and side =="MSP":
            
            df_g = msp.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options9 == "College Encouraged me to take up this course" and side =="APT":
            
            df_g = apt.groupby(['College_Encouragement', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['College_Encouragement', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['College_Encouragement', 'Branch', 'Counts', 'Percentage']
            
            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
#--------------------overall evaluation---------------------------------    
with tab4:
    
    
    st.header("Overall Evaluation")
    options6=st.selectbox('Select the Question', ["I would highly recommend this course to other students", "I would highly recommend this instructor to other students", "The course is helpful in progress toward my career", 
"This course has a high educational value and impact", "This course was challenging", "Overall, this course is meeting my expectations for the quality of the course", "Would you be interested in such Programs in the Future?", "How would you describe your experience?"], key="options6")
    
     #------------Recommend_Course---------
        
    if options6 == "I would highly recommend this course to other students" and side == "All":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = df.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)
        
            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this course to other students" and side == "SAK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sak.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "I would highly recommend this course to other students" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sat.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']            

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this course to other students" and side == "SRS":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = srs.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "I would highly recommend this course to other students" and side == "NKT":
        
        

            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = nkt.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 =="I would highly recommend this course to other students" and side == "RGT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = rgt.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage'] 

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "I would highly recommend this course to other students" and side == "SHT":

        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sht.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this course to other students" and side == "SSK":
        
        
          
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = ssk.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "I would highly recommend this course to other students" and side == "MSP":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = msp.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this course to other students" and side == "APT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = apt.groupby(['Recommend_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Recommend_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
 #------------Recommend_Instructor---------
        
    if options6 == "I would highly recommend this instructor to other students" and side == "All":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = df.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "SAK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sak.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sat.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "SRS":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = srs.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "NKT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = nkt.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "RGT":
        
          
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = rgt.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "SHT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
        
        
            df_g = sht.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options6 == "I would highly recommend this instructor to other students" and side == "SSK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree') 
        
        
            df_g = ssk.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']    

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "I would highly recommend this instructor to other students"and side == "MSP":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = msp.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "I would highly recommend this instructor to other students" and side == "APT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = apt.groupby(['Recommend_Instructor', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Recommend_Instructor', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Recommend_Instructor', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
#------------Course_helpful---------
        
    if options6 == "The course is helpful in progress toward my career" and side == "All":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = df.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "The course is helpful in progress toward my career" and side == "SAK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sak.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "The course is helpful in progress toward my career" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sat.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "The course is helpful in progress toward my career" and side == "SRS":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = srs.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "The course is helpful in progress toward my career" and side == "NKT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = nkt.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "The course is helpful in progress toward my career" and side == "RGT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = rgt.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "The course is helpful in progress toward my career" and side == "SHT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')


            df_g = sht.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage'] 

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "The course is helpful in progress toward my career" and side == "SSK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree') 


            df_g = ssk.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']            

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "The course is helpful in progress toward my career"and side == "MSP":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = msp.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "The course is helpful in progress toward my career" and side == "APT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            fig = px.histogram(apt, x='Course_helpful', color='Branch', barmode = 'group', text_auto=True)
            df_g = apt.groupby(['Course_helpful', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Course_helpful', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_helpful', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
#------------Course_Impact---------
        
    if options6 == "This course has a high educational value and impact" and side == "All":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = df.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course has a high educational value and impact" and side == "SAK":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sak.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']
            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "This course has a high educational value and impact" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            fig = px.histogram(sat, x='Course_Impact', color='Branch', barmode = 'group', text_auto=True)
            df_g = sat.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course has a high educational value and impact" and side == "SRS":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = srs.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "This course has a high educational value and impact" and side == "NKT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = nkt.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course has a high educational value and impact" and side == "RGT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = rgt.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
            
    elif options6 == "This course has a high educational value and impact" and side == "SHT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sht.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
        
    elif options6 == "This course has a high educational value and impact" and side == "SSK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = ssk.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )      
            
    elif options6 == "This course has a high educational value and impact"and side == "MSP":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = msp.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )      
    elif options6 == "This course has a high educational value and impact" and side == "APT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = apt.groupby(['Course_Impact', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Course_Impact', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Course_Impact', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )    
#------------Challenging---------
        
    if options6 == "This course was challenging" and side == "All":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = df.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )   
    elif options6 == "This course was challenging" and side == "SAK":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sak.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']  

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "This course was challenging" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sat.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course was challenging" and side == "SRS":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = srs.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
        
    elif options6 == "This course was challenging" and side == "NKT":
        
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = nkt.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage'] 

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
    elif options6 == "This course was challenging" and side == "RGT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = rgt.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "This course was challenging" and side == "SHT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sht.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course was challenging" and side == "SSK":
        
        
          
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = ssk.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage'] 

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "This course was challenging"and side == "MSP":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = msp.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "This course was challenging" and side == "APT":
        
       
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = apt.groupby(['Challenging', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Challenging', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Challenging', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
#-----------Quality_Course---------
        
    if options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "All":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = df.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "SAK":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = sak.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "SAT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sat.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']
            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "SRS":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = srs.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']            

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "NKT":
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = nkt.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "RGT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = rgt.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "SHT":
        
        
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')
            df_g = sht.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "SSK":
        
        
          
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = ssk.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course"and side == "MSP":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = msp.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Overall, this course is meeting my expectations for the quality of the course" and side == "APT":
        
        
            st.markdown('1-Strongly disagree 2-Disagree 3-Neutral 4-Agree 5-Strongly Agree')

            df_g = apt.groupby(['Quality_Course', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Quality_Course', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Quality_Course', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
#-----------Future_interest---------
        
    if options6 == "Would you be interested in such Programs in the Future?" and side == "All":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = df.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "SAK":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = sak.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
            
            
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "SAT":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = sat.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']            

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "SRS":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = srs.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']  

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "NKT":
        
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = nkt.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "RGT":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = rgt.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "SHT":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')


            df_g = sht.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "SSK":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely') 


            df_g = ssk.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "MSP":
        
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = msp.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "Would you be interested in such Programs in the Future?" and side == "APT":
        
        
            st.markdown('1-Extremely Likely 2-Unlikely 3-Neutral 4-Likely 5-Extremely Likely')

            df_g = apt.groupby(['Future_interest', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Future_interest', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Future_interest', 'Branch', 'Counts', 'Percentage']
            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
#-----------Experience---------
        
    if options6 == "How would you describe your experience?" and side == "All":
        
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = df.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "How would you describe your experience?" and side == "SAK":
        
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = sak.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "SAT":
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = sat.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "How would you describe your experience?" and side == "SRS":
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = srs.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "NKT":
        
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = nkt.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage'] 

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "RGT":
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = rgt.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "SHT":
        
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')


            df_g = sht.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "SSK":
        
        
          
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = ssk.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
    elif options6 == "How would you describe your experience?" and side == "MSP":
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = msp.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options6 == "How would you describe your experience?" and side == "APT":
        
        
            st.markdown('1-Very Unsatisfied 2-Unsatisfied 3-Neutral 4-Satisfied 5-Satisfied')

            df_g = apt.groupby(['Experience', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Experience', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Experience', 'Branch', 'Counts', 'Percentage']  

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
#-----------------overall score---------------   
with tab5:
    st.header("Overall Score")
    options7=st.selectbox('Select the Question', ["On a scale of 1-10 how would rate this program"], key="options7")
    
    #-----------Score---------
        
    if options7 == "On a scale of 1-10 how would rate this program" and side == "All":
        
            st.markdown('1 = Poor, 10  =  Good')



            df_g = df.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = df.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "SAK":
        
            st.markdown('1 = Poor, 10  =  Good')


            df_g = sak.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = sak.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "SAT":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = sat.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = sat.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "SRS":
        
            st.markdown('1 = Poor, 10  =  Good')


            df_g = srs.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = srs.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "NKT":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = nkt.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = nkt.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
            
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "RGT":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = rgt.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = rgt.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']  

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )  
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "SHT":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = sht.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = sht.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "SSK":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = ssk.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = ssk.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   ) 
            
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "MSP":
        
            st.markdown('1 = Poor, 10  =  Good')

            df_g = msp.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = msp.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']            

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )
            
    elif options7 == "On a scale of 1-10 how would rate this program" and side == "APT":
        
            st.markdown('1 = Poor, 10  =  Good')



            df_g = apt.groupby(['Score', 'Branch']).size().reset_index()
            df_g['percentage'] = apt.groupby(['Score', 'Branch']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
            df_g.columns = ['Score', 'Branch', 'Counts', 'Percentage']

            st.table(df_g)

            @st.cache
            def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df_g)

            st.download_button(
                    label="Download",
                    data=csv,
                    file_name='df_g.csv',
                    mime='text/csv',
                   )    
            
            
            
            
            
            
            
            
            

            
            
            
        

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    