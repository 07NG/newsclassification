from email import message
from enum import unique
from operator import mod
from this import d
from tokenize import String
from turtle import title
from flask import Flask, render_template, redirect, request, url_for, jsonify
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from matplotlib.style import context
from sqlalchemy import CHAR, desc
from wtforms import StringField, PasswordField, BooleanField, SelectField
from wtforms import validators
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from newsapi.newsapi_client import NewsApiClient
import pickle
import pandas as pd
import email_validator
from forms import LoginForm,RegisterForm,UploadNewsForm


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
tfid = pickle.load(open('feature.pkl','rb'))

app.config['SECRET_KEY']='Thhisissecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\Project\database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)    
login_manager =  LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

class News(db.Model):
    title = db.Column(db.String(250), primary_key=True)
    desc = db.Column(db.String(600))
    img = db.Column(db.String(200))
    content = db.Column(db.String(1500))
    url =  db.Column(db.String(200))
    ctr = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = StringField('password', validators=[InputRequired(), Length(min= 8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid Email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

class UploadNewsForm(FlaskForm):
    title = StringField('Title', validators=[InputRequired(), Length(min=10, max=200)])
    desc = StringField('Description', validators=[InputRequired(), Length(min=10, max=2000)])


def GetNewsOnline(source,ctr):
    newsapi = NewsApiClient(api_key="38122ac1faf54ee2acdbc704e062cd89")
    topheadlines = newsapi.get_top_headlines(sources=source, category=ctr, page_size=90,language='en')

    articles = topheadlines['articles']

    desc = []
    news = []
    img = []
    url = []

    for i in range (len(articles)):
        myarticles = articles[i]
        news.append(myarticles['title'])
        desc.append(myarticles['description'])
        img.append(myarticles['urlToImage'])
        url.append(myarticles['url'])
        
        news_title = News.query.filter_by(title=myarticles['title']).first()
        if news_title:
            None
        else:
            new_news = News(title = myarticles['title'], desc = myarticles['description'], img = myarticles['urlToImage'], content = myarticles['content'], url = myarticles['url'], ctr = ctr)
            db.session.add(new_news)
            db.session.commit()

    myList = zip(news, desc, img, url)    
    return myList


def GetNews(source,ctr=None):

    db_news = News.query.all()

    desc = []
    news = []
    img = []
    url = []

    if ctr:
        for n in reversed(db_news):
            if n.ctr == ctr:
                news.append(n.title)
                desc.append(n.desc)
                img.append(n.img)
                url.append(n.url)
    else:
        for n in reversed(db_news):
            news.append(n.title)
            desc.append(n.desc)
            img.append(n.img)
            url.append(n.url)

    myList = zip(news, desc, img, url)    
    return myList

@app.route('/')
def index():

    #new_news = News(id=22,title="RAM")
   #db.session.add(new_news)
    #db.session.commit()

    
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1> Invalid username or password </h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET','POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route('/dashboard',methods=['GET','POST'])
@login_required
def dashboard():
    
    dash = GetNews(source=None, ctr="technology")

    return render_template('dashboard.html', name=current_user.username, context=dash)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/business')
@login_required
def business():

    business = GetNews(source=None, ctr="business")

    return render_template('business.html', name=current_user.username, context=business)

@app.route('/health')
@login_required
def health():

    health = GetNews(source=None, ctr="health")

    return render_template('health.html', name=current_user.username, context=health)

@app.route('/entertainment')
@login_required
def entertainment():

    entertainment = GetNews(source=None, ctr="entertainment")

    return render_template('entertainment.html', name=current_user.username, context=entertainment)

@app.route('/sports')
@login_required
def sports():

    sports = GetNews(source=None, ctr="sports")

    return render_template('sports.html', name=current_user.username, context=sports)

@app.route('/technology')
@login_required
def technology():

    technology = GetNews(source=None, ctr="technology")

    return render_template('technology.html', name=current_user.username, context=technology)

@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    form = UploadNewsForm()

    if form.validate_on_submit():

        # input = form.desc.data
        # input = [input]
        
        # new_news = tfid.transform(input).toarray()
        # prediction = model.predict(new_news)
        # print(prediction)

        # print("here")
        news_title = News.query.filter_by(title=form.title.data).first()
        if news_title:
            None
        else:
            new_news = News(title = form.title.data, desc = form.desc.data, img = None, content = None, url = None, ctr = None)
            db.session.add(new_news)
            db.session.commit()
        
        return redirect(url_for('processing', title = form.title.data, desc= form.desc.data))

    return render_template('upload.html', form=form)

@app.route('/processing/<title>/<desc>', methods=['GET','POST'])
@login_required
def processing(title, desc):
    
    input = [desc]
    new_news = tfid.transform(input).toarray()
    prediction = model.predict(new_news)
    print(prediction)

    if prediction == 0:
        context = 'Business'
    elif prediction == 1:
        context = 'Entertainment'
    elif prediction == 2:
        context = 'Politics'
    elif prediction == 3:
        context = 'Sports'
    else:
        context = 'Technology'


    return render_template('processing.html', t = title, d = desc, context=context)

@app.route('/dbnews')
@login_required
def dbnews():

    health = GetNewsOnline(source=None,ctr='health')
    print("done")
    business = GetNewsOnline(source=None,ctr='business')
    print("done")
    sports = GetNewsOnline(source=None,ctr='sports')
    print("done")
    entertainment = GetNewsOnline(source=None,ctr='entertainment')
    print("done")
    technology = GetNewsOnline(source=None,ctr='technology')
    print("done")
    
    return render_template('dbnews.html', name=current_user.username, context=[health,business,sports,technology,entertainment])

@app.route('/guest')
def guest():

    guest = GetNews(source=None,ctr=None)

    return render_template('guest.html',context=guest)

if __name__ == '__main__':
    app.run(debug=True)