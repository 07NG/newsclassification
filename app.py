import pickle

import nltk
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from newsapi.newsapi_client import NewsApiClient
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from model import Accuracy
from forms import LoginForm, RegisterForm, UploadNewsForm

nltk.download('stopwords')

nltk.download('punkt')

import warnings

warnings.filterwarnings("ignore")

app = Flask(
    __name__)  # create instance of flask web application using flask constructor __name__ tells name of application to locate other files and templates
model = pickle.load(open('model.pkl', 'rb'))  # loads machine learning model from binary file ie,model.pkl
tfid = pickle.load(open('feature.pkl', 'rb'))  # loads feature extraction model from binary file i.e,feature.pkl

app.config[
    'SECRET_KEY'] = 'Thhisissecretkey'  # sets 'secret key' config to flask application instance i.e, app to 'Thhisissecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\Project\database.db'  # Configure db connection
bootstrap = Bootstrap(app)  # initalize bootstrap extension to app
db = SQLAlchemy(app)  # intialize SQLAlchemy extension to app
login_manager = LoginManager()  # initialize login manage() for authentication,authorization and user session
login_manager.init_app(app)  # Registers login manager extenstion with app
login_manager.login_view = 'login'  # sets login_manager.login_view to login


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
    url = db.Column(db.String(200))
    ctr = db.Column(db.String(20))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid Email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])


class UploadNewsForm(FlaskForm):
    title = StringField('Title', validators=[InputRequired(), Length(min=10, max=200)])
    desc = StringField('Description', validators=[InputRequired(), Length(min=24, max=2000)])


def GetNewsOnline(source, ctr):  # defining function
    newsapi = NewsApiClient(
        api_key="38122ac1faf54ee2acdbc704e062cd89")  # assigning newsapi with newsapiclient(api key)->api key is req authenticate and acces newsapi service
    topheadlines = newsapi.get_top_headlines()  # acess news with top headlines

    articles = topheadlines['articles']  # assigning articles with articles with topheadlines

    desc = []  # assigning list
    news = []
    img = []
    url = []

    for i in range(len(articles)):
        myarticles = articles[i]
        news.append(myarticles['title'])
        desc.append(myarticles['description'])
        img.append(myarticles['urlToImage'])
        url.append(myarticles['url'])

        news_title = News.query.filter_by(title=myarticles[
            'title']).first()  # Querying 'News' table and retrieving title with first row that matches
        if news_title:  # Checks if news_title is specified
            None
        else:  # if title not specified
            new_news = News(title=myarticles['title'], desc=myarticles['description'], img=myarticles['urlToImage'],
                            content=myarticles['content'], url=myarticles['url'], ctr=ctr)
            # specifying news details
            db.session.add(new_news)  # add new news to db
            db.session.commit()  # save changes into db

    myList = zip(news, desc, img, url)  # Zips 4 list into list of tuples repr news,desc,img,url
    return myList  # returns list of tuples


def GetNews(source, ctr=None):  # defininf function GetNews with source and category none

    db_news = News.query.all()  # Query 'News' table and retireve all

    desc = []  # Assigning desc to list
    news = []
    img = []
    url = []

    if ctr:  # checks if category is specified
        for n in reversed(db_news):  # from newset to oldest news from database
            if n.ctr == ctr:  # Checks if current news articles belongs to specified news or not
                news.append(n.title)  # if belongs to category then append title,desc,image and url
                desc.append(n.desc)
                img.append(n.img)
                url.append(n.url)
    else:  # if category not specified
        for n in reversed(db_news):  # from newest to oldest
            news.append(n.title)  # append title,desc,img and url
            desc.append(n.desc)
            img.append(n.img)
            url.append(n.url)

    myList = zip(news, desc, img, url)  # Zips 4 list into list of tuples repr news,desc,img,url
    return myList  # returns list of tuples


@app.route('/')  # Register view function
def index():  # View function that process incoming req,perform and reponse

    # new_news = News(id=22,title="RAM")
    # db.session.add(new_news)
    # db.session.commit()

    return render_template('index.html')  # Renders index.html template


@app.route('/login', methods=['GET', 'POST'])  # Register view function with either GET or POST method
def login():  # View function that process incoming req,perform and reponse
    form = LoginForm()  # LoginForm instanste creation and assigns to form

    if form.validate_on_submit():
        user = User.query.filter_by(
            username=form.username.data).first()  # Query 'User' table in db and checks username into db and retrieve first row else none
        if user:  # Checks if user exists or not and user will be set to corresponding 'User' object
            if check_password_hash(user.password, form.password.data):  # Check if hashed password is matched or not
                login_user(user,
                           remember=form.remember.data)  # Log in user with user object and remember which will save if browser is closed too
                return redirect(url_for('dashboard'))  # Redirects to dashboard view function

        return '<h1> Invalid username or password </h1>'

    return render_template('login.html', form=form)  # Renders login.html with passing form=Loginform()


@app.route('/signup', methods=['GET', 'POST'])  # Register view function with either GET or POST method
def signup():  # View function that process incoming req,perform and reponse
    form = RegisterForm()  # RegisterForm instanste creation and assigns to form

    if form.validate_on_submit():  # Check form validation from RegisterForm defined in forms.py
        hashed_password = generate_password_hash(form.password.data, method='sha256')  # Password hashed with SHA256
        new_user = User(username=form.username.data, email=form.email.data,
                        password=hashed_password)  # new_user instance with User info
        db.session.add(new_user)  # Add new_user to database
        db.session.commit()  # Save changes to database

        return redirect(url_for('login'))  # Redirects to login page with login view function after sucessful operation

    return render_template('signup.html',
                           form=form)  # Renders signup.html and passing form to signup.html help to render correct field and validate which is defined in forms.py


@app.route('/dashboard', methods=['GET', 'POST'])  # Registers view function and response with either GET OR POST method
@login_required  # Access to logged in user otherwise redirect to login page
def dashboard():  # defining view function that process,performs and response

    dash = GetNews(source=dbnews)  # GetNews class is reponsible for retrieving news article form news API or source,
    # here source is None so retrieve form newsAPI and ctr is for category,here it is Technology category
    return render_template('dashboard.html', name=current_user.username,
                           context=dash)  # Renders dashboard.html with current username and


@app.route('/logout')  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def logout():  # defining view function
    logout_user()  # End session of current user
    return redirect(url_for('index'))  # Redirects to index view function


@app.route('/business', methods=['GET'])  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def business():  # defining view function

    business = GetNews(source=dbnews,
                       ctr="business")  # GetNews class is reponsible for retrieving news article form news API or source,
    # here source is None so retrieve form newsAPI and ctr is for category,here it is Business category

    return render_template('business.html', name=current_user.username,
                           context=business)  # Renders businees.html template with current username and category


@app.route('/entertainment')  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def entertainment():  # defining view function

    entertainment = GetNews(source=dbnews, ctr="entertainment")

    return render_template('entertainment.html', name=current_user.username, context=entertainment)


@app.route('/politics')  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def politics():  # defining view function

    politics = GetNews(source=dbnews, ctr="politics")

    return render_template('politics.html', name=current_user.username, context=politics)


@app.route('/sports')  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def sports():  # defining view function

    sports = GetNews(source=dbnews, ctr="sports")

    return render_template('sports.html', name=current_user.username, context=sports)


@app.route('/technology')  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def technology():  # defining view function

    technology = GetNews(source=dbnews, ctr="technology")

    return render_template('technology.html', name=current_user.username, context=technology)


@app.route('/upload', methods=['GET', 'POST'])  # Registers view function
@login_required  # Access restriction to logged in  user and if not they are redirected to login form
def upload():  # defining view function
    form = UploadNewsForm()  # Assigning UploadNewsForm class in form

    if form.validate_on_submit():  # Checking for form validation on submit

        input = form.desc.data
        input = [input]

        newnews = tfid.transform(input).toarray()
        prediction = model.predict(newnews)
        if prediction == 0:
            context = 'business'
        elif prediction == 1:
            context = 'entertainment'
        elif prediction == 2:
            context = 'politics'
        elif prediction == 3:
            context = 'sports'
        else:
            context = 'technology'

            print(context)
        news_title = News.query.filter_by(
            title=form.title.data).first()  # Quering the 'News' table in database and retrieving title whose matches
        # form title and first() retrieve first row that matches filter if none the 'None' is returened
        if news_title:
            None
        else:
            new_news = News(title=form.title.data, desc=form.desc.data, img=None, content=None, url=None, ctr=context)
            db.session.add(new_news)  # Add new_news to the db
            db.session.commit()  # Saves changes to db

        return redirect(url_for('processing', title=form.title.data, desc=form.desc.data))

    return render_template('upload.html', form=form)


@app.route('/processing/<title>/<desc>', methods=['GET', 'POST'])  # Resigisters view function
@login_required  # Access priviledge for logged in user else redirect to login page
def processing(title, desc):  # defining view function with title and desc as parameter for processing

    input = [desc]  # assigns input text with list of strings
    new_news = tfid.transform(input).toarray()  # transform(input) changes input into matrix of numeric features
    # and toarray changes martix into numpy array
    prediction = model.predict(new_news)  # Predicts new_news with pre-trained model which is trained in model.py
    print(prediction)
    accuracy = Accuracy()
    print (accuracy)
    if prediction == 0:
        context = 'Business'

    elif prediction == 1:
        context = 'Entertainment'

    elif prediction == 2:
        context = 'Politics'

    elif prediction == 3:
        context = 'Sports '

    else:
        context = 'Technology'

    return render_template('processing.html', t=title, d=desc, context=context, accuracy=accuracy)


@app.route('/dbnews', methods=['GET', 'POST'])  # Regiseter view function
@login_required  # Access priviledge for logged in user if not redirect to login page
def dbnews():  # defining view function

    business = GetNewsOnline(source=None, ctr='business')
    print("done")
    entertainment = GetNewsOnline(source=None, ctr='entertainment')
    print("done")
    politics = GetNewsOnline(source=None, ctr='politics')
    print("done")
    sports = GetNewsOnline(source=None, ctr='sport')
    print("done")
    technology = GetNewsOnline(source=None, ctr='technology')
    print("done")

    return render_template('dbnews.html', name=current_user.username,
                           context=[politics, business, sports, technology, entertainment])


@app.route('/guest')  # Register view function
def guest():  # defining view function

    guest = GetNews(source=dbnews)  # Getting news from newsApi as source is None and caterogry is none too

    return render_template('guest.html', context=guest)  # Renders guest.html and passed guest from above


if __name__ == '__main__':  # Checks whether current module is run as main module
    app.run(debug=True)  # Resposnible for running the app in debug mode
