const express = require('express');
const app = express();
const port = 3000;
const path = require('path');

// Set up a route that returns a list of pictures in the database
app.get('/pictures', (req, res) => {
  // Replace this with code that retrieves the list of pictures from your database
  const pictures = ['6_MetroTech.png', 'floorplan.png'];
  res.send(pictures);
});

// Set up a route that serves a specific picture from the database
app.get('/pictures/:id', (req, res) => {
  // Replace this with code that retrieves the picture from your database based on the ID
  const picture = 'picture' + req.params.id + '.png';
  // Serve the picture file using Express's built-in sendFile method
  res.sendFile(path.join(__dirname, 'pictures', picture));
});

// Set up a route that allows users to upload pictures to the database
app.post('/pictures', (req, res) => {
  // Replace this with code that handles the file upload and stores the picture in your database
  res.send('Picture uploaded successfully!');
});

// Start the server and listen for incoming requests
app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});

var fs = require("fs");
var multer  = require('multer');
app.use('/public',express.static('public'));
app.use(multer({ dest: '/tmp/'}).array('image'));
app.post('/file_upload',function (req, res) {
    fs.readFile( req.files[0].path, function (err, data) {
        var des_file = __dirname + "/public/img/" + req.files[0].originalname;
        fs.writeFile(des_file, data, function (err) {
            if(err){
                console.log( err );
            }else{
               var response = {
                    message:'Image input success',
                    filename:req.files[0].originalname
                };
            }
            res.send( JSON.stringify( response ) );
        });
    });
})
var server = app.listen(8081, function () {
    var host = server.address().address;
    var port = server.address().port;
    console.log("Image address: http://", host, port)
})
//Upload picture database


var fs = require("fs");
var path = require('path');
var uuid = require('uuid');
var formidable = require('formidable');
app.use('/public',express.static('public'));
app.post('/file_upload',function (req, res) {
    var form = new formidable.IncomingForm();
    //Create form
    form.encoding = 'utf-8';
    //Set up encoding
    form.uploadDir = "public/img";
    //Set up directory
    form.keepExtensions = true;
    //Extension
    form.maxFieldsSize = 2 * 1024 * 1024;
    //Field size
    form.parse(req, function (err, fields, files) {
        var file = files.files;
        let picName = uuid.v1() + path.extname(file.name);
        fs.rename(file.path, 'public\\img\\' + picName, function (err) {
            if (err) return res.send({ "error": 403, "message": "Image error!" });
            res.send({ "picAddr": picName });
        });
    });
})
var server = app.listen(8081, function () {
    var host = server.address().address;
    var port = server.address().port;
    console.log("Server listening at: http://", host, port)
})