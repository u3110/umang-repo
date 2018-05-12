from flask import Flask, render_template, request, json
from scramble import mainF
from Encryption import ymain
from zig import zmain 
import os
import subprocess
#from programm import rsamain
app = Flask(__name__)
import pandas as pd

@app.route("/")
def main():
    return render_template('main.html')

@app.route('/srun', methods = ['POST'])
def imgIn():
    _imgin = request.form["img_loc"]
    img_d = mainF(_imgin)
    img_yd = ymain(_imgin)
    img_zd = zmain(_imgin)
    #img_rsad = rsamain(_imgin)
    #img=""

    #command = ["sudo python3 ","/home/ubuntu/Desktop/major/programm.py ",_imgin]
    #img_rsad = subprocess.check_output(command,_imgin)
    img_rsad = os.system("sudo python3 programm.py "+str(_imgin))
    #print img_rsad.decode("utf-8")
    print img_rsad
    #print img_d
    #if img_rsad:
    try:
        rsa_out = open("output.csv",'r')
        img1=rsa_out.read()
	img =  img1.split(',')
        rsa_out.close()
        return render_template('output.html', img_d = img_d, img_yd = img_yd, img_zd = img_zd, img_rsad = img)
    except Exception as ee:
        return "try later"


    # imgin = "/Users/yash/PycharmProjects/Major/test.jpg"
    # print imgin
    # mainF('/Users/yash/PycharmProjects/Major/test.jpg')
    # return render_template('scrambled.html', imgin = imgin)

    # return render_template('test2.html', name= imgin)

@app.route('/test2/<name>')
def showSignUp(name):
    return render_template('test2.html', name = name)


if __name__ == "__main__":
    app.run(debug=True,port=8000)
