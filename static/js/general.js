/* This is your general javascript, it has been embedded into layout.html. 
Any scripts here are available through out the entire website */

exit_button.onclick = exit;
function exit() {
    //alert("Shutting Down")
    new_ajax_helper('/exit'); 
  };