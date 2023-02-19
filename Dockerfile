FROM python
WORKDIR /pythondir
COPY . /pythondir
EXPOSE 8800
RUN pip install -r requirements.txt
CMD streamlit run server.py