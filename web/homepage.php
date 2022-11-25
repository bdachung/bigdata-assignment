<?php
    include_once("connection.php");
?>

<!DOCTYPE html>
<html>
    <head>
        <title>Sentiment Analysis</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.6.1/css/all.css" integrity="sha384-gfdkjb5BdAXd+lj+gudLWI+BXq4IuLW5IT+brZEZsLFm++aCMlF1V92rMkPaX4PP" crossorigin="anonymous">
        
        <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
        <script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        <link rel="stylesheet" href="style.css">
    </head>

    <body>
        <div id="container">
            <h1>Sentiment Analysis</h1>
            <form id="f">
                <label for="sentence">Enter your sentence</label></br>
                <textarea name="sentence" id="sentence" form="f" placeholder="Please enter here" cols="100" rows="10"></textarea>
</br>
                <button type="button" class="btn btn-primary" onclick="classifySentence()" style="margin-right: 650px;">Predict</button>
            </form>
            <p style="color : blue ;">Result:</p>
            <p style="color: red ; " id="result"></p>
        </div>

        <script>
            function classifySentence(){
                let sen = document.getElementById("sentence").value;
                console.log(sen);
                let xmlhttp = new XMLHttpRequest();
                xmlhttp.onreadystatechange = function() {
                    if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
                        let data = xmlhttp.responseText;
                        data = JSON.parse("{" + xmlhttp.responseText.slice(1,-1).replaceAll('\\',"") + "}");
                        let result = document.getElementById("result");
                        result.innerHTML = "Logistic: " + data['LOR'] + "</br>Random Forest: " + data['RF'] + "</br>Gradient Boosting " + data['GB'] + "</br>phoBert: " + data['PB'];
                    }
                }
                xmlhttp.open("POST", "<?php echo $AISERVER; ?>" + "/predict");
                xmlhttp.setRequestHeader("Content-type", "application/json");
                xmlhttp.send(JSON.stringify({"sen":sen}));
            }
        </script>
    </body>
</html>