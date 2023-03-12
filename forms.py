from wtforms import StringField, PasswordField, BooleanField, SelectField
from wtforms import validators
from wtforms.validators import InputRequired, Email, Length
from flask_wtf import FlaskForm

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