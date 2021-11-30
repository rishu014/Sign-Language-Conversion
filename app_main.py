from flask import Flask
from flask import render_template
from flask import  Blueprint
import jyserver.Flask as jsf 

from cam import show, speak
from cam import saveFile
from cam import clear
import time
import os



app = Flask(__name__)

@jsf.use(app)
class App:
    def __init__(self):
        self.mess=''
        self.audio=''
        
    def data(self):
        self.mess=open('STORAGE.txt')
        self.show=self.mess.read()
        self.js.document.getElementById('scrip').innerHTML='<p>'+self.show+'</p>'   
        self.js.document.getElementById('scrip').style.display='block'
   
@app.route('/')
def home():
    return App.render(render_template('home.html'))

@app.route('/camera_feed')
def cam():
    show()
    return App.render(render_template('home.html'))

@app.route('/saving', endpoint='saveFile')
def  cam():
    saveFile()
    return App.render(render_template('home.html'))


@app.route('/clear',endpoint='clear')
def cam():
    clear()
    return App.render(render_template('home.html'))


@app.route('/speaks',endpoint='speak')
def cam():
    speak()
    return App.render (render_template('home.html'))


@app.route('/about',endpoint="about")
def about():
    return render_template('about.html')

@app.route('/developers',endpoint='developers')
def developer():
    return render_template('developers.html')

@app.route('/contact',endpoint='contact')
def about():
    return render_template('contactUs.html')






if __name__ == '__main__':
    app.run(debug=True)
