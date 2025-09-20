from flask import Flask, request, jsonify, flash, redirect, url_for, Blueprint
from flask_restful import Resource, Api
from flask_uploads import UploadSet, configure_uploads, IMAGES, DOCUMENTS, ALL
from newfolder.DOCU_RAG import d_rag
# from werkzeug.utils

allowed_extension={'txt', 'pdf', 'png', 'jpg'}
app=Flask(__name__)
api=Api(app)

app.config['UPLOADED_FILES_DEST']='uploads'

files=UploadSet('files',allowed_extension)
configure_uploads(app,files)

class Home(Resource):
    def get(self):
        return "hello welcome"

class UploadFile(Resource):
   def post(self):
       file=request.files['file']
       if 'file' not in request.files:
           return {"error":"No file selected"}

       if file.filename=='':
           return {"error":"No file selected"}

       try:
           filename=files.save(file)
           file_path=files.path(filename)
           return {
               "message":"File uploaded successfully",
               "filename":filename,
               "filepath":file_path
           },201
       except Exception as e:
           return {"error": str(e)}, 500


api.add_resource(UploadFile,'/upload')
api.add_resource(Home,'/')
app.register_blueprint(d_rag,url_prefix='/gett')

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0", port=5004)


