
//This ajax helper caters for files and does not require JQuery, an example of its use can be seen below:
/*
  In pythonanywhere you will need to link it within the HTML: <script src='/static/js/new_ajax_helper.js'></script>
  There are two ways of creating a form object - easiest is below:
  testform = document.getElementById('testform'); //get the form element

  var formobject = new FormData(testform); //create a form object since one does not exist
  formobject.append("testinput", testinput); //to add an individual variable use a key and value
  Another means to just to create one from scratch:
  formobject = new FormData(); //create a form object
  formobject.append("email", email); //email is a textinput tag value
  formdata.append('file',files[0]); //file is a filesinput tag value

  new_ajax_helper('/test',defaulthandler,formobject); //send the formobject to the url, you can define a callback
*/

function new_ajax_helper(url, callback=defaulthandler, formobject=null, method='POST')
{
    //create a request object
    var xhr = new XMLHttpRequest();

    xhr.open(method, url, true);
    xhr.onreadystatechange = function() //callback function
    {
        if (xhr.readyState == 4) //4 means data received
        {
            results = JSON.parse(xhr.responseText); //change JSON into a Javascript object
            callback(results); //call the callback function
        }
    }
    xhr.send(formobject); //send the form data
}

function defaulthandler(results)
{
    console.log(results);
}