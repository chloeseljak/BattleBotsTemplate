FROM python:3

RUN pip install requests
RUN pip install pydantic
RUN pip install nltk 
RUN pip install openai
RUN pip install numpy
RUN pip install xgboost
RUN pip install sentence_transformers
RUN pip install joblib 
RUN pip install uuid
#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
