import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from PIL import Image

image = Image.open('story.png')



page_title = "Story Maker"
st.set_page_config(
    page_title=page_title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "Demo Page by AdCreativeDEv"
    }
)
st.image(image, caption='created by midjourney ')
st.title(":blue[" + page_title + "]")


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Azure Open AI version
# azllm=AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003", temperature=0.5)

st.subheader(':robot_face: I will generate :blue[Synopsis] , please enter :green[title] and :green[era] as below:')

col1, col2  = st.columns(2)
with col1:
    user_title = st.text_input(":point_right: title ", value="a fat mouse finding chesse at lazy cat living room")
    user_era = st.text_input(":point_right: era ", value="Future 2046")

with col2:
    # Set API keys
    system_openai_api_key = os.environ.get('OPENAI_API_KEY')
    system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
    os.environ["OPENAI_API_KEY"] = system_openai_api_key

    

llm = OpenAI(
          model_name="text-davinci-003", # default model
          temperature=0.9,
          verbose=True) 


if st.button("Create"):
    st.subheader(":chains: Chain 1 - Synopsis chain")
   

    # This is an LLMChain to write a synopsis given a title of a play.
    template = """You are a playwright. Given the title of play and the era it is set in, 
                    it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""
    prompt_template = PromptTemplate(input_variables=["title", 'era'], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis", verbose=True)
    st.info(synopsis_chain.prompt)
    



    ############################
    # Chain2 - write a Review of this Synopsis
    ############################
    st.subheader(":chains: Chain 2 - Play Review Chain")
    # This is an LLMChain to write a review of a play given a synopsis.
    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", verbose=True)

    st.info(review_chain.prompt)


    st.subheader(":book: Synopsis and Review")
    overall_chain = SequentialChain(
            chains=[synopsis_chain, review_chain],
            input_variables=["era", "title"],
            output_variables=["synopsis", "review"],
            verbose=True)
            
    
    
    output = overall_chain({"title": user_title, "era": user_era})
    st.success(output, icon="✅")
    
    

        
    # This is the overall chain where we run these two chains in sequence.
    


log = """
> Entering new SequentialChain chain...


> Entering new LLMChain chain...
Prompt after formatting:
You are a playwright. Given the title of play and the era it is set in, 
                    it is your job to write a synopsis for that title.

    Title: a fat mouse finding chesse at lazy cat living room
    Era: Future 2046
    Playwright: This is a synopsis for the above play:

> Finished chain.


> Entering new LLMChain chain...
Prompt after formatting:
You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    

In a future world of 2046, a fat mouse named Findley is determined to find the ultimate supply of cheese. After a long and arduous journey, he finds what he's been seeking - a home in the living room of the lazy cat, Bumbles. As Findley explores his newfound haven, he quickly discovers that the cat isn't too fond of the idea of having a fat mouse in her home. A series of comedic and heartwarming encounters ensues as Findley attempts to outwit Bumbles and find a way to peacefully live together in the living room. Through their adventures, Findley and Bumbles learn to set aside their differences and become unlikely friends. In the end, the two are able to find a middle ground, allowing Findley and his family to stay in peace and feast on the endless supply of cheese.
    Review from a New York Times play critic of the above play:

> Finished chain.


################################
# code
################################
  

    # This is an LLMChain to write a synopsis given a title of a play.
    template = 'You are a playwright. Given the title of play and the era it is set in, 
                    it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:'
    prompt_template = PromptTemplate(input_variables=["title", 'era'], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis", verbose=True)
    st.info(synopsis_chain.prompt)
    



    ############################
    # Chain2 - write a Review of this Synopsis
    ############################
    st.subheader(":chains: Chain 2 - Play Review Chain")
    # This is an LLMChain to write a review of a play given a synopsis.
    template = 'You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:'
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", verbose=True)

    st.info(review_chain.prompt)


    st.subheader(":book: Synopsis and Review")
    overall_chain = SequentialChain(
            chains=[synopsis_chain, review_chain],
            input_variables=["era", "title"],
            output_variables=["synopsis", "review"],
            verbose=True)
            
    
    
    output = overall_chain({"title": user_title, "era": user_era})
    st.success(output, icon="✅")
    



"""
with st.expander("explanation"):
    st.code(log)