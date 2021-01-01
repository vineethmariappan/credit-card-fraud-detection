var json_obj=[];
var json;
$(document).on('submit', '#fm', (e) => {

    e.preventDefault();
    $("#result").removeClass();
    console.log(json);
    document.getElementById("result").innerHTML = '';
    $('#loading').append(`  
    <div class="spinner-border" role="status">
    <span class="sr-only">Loading...</span>
    </div>
    `)
    $.ajax({
        type: 'POST',
        url: 'home',
        data: {
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
            transaction : json
        },
        success: function (data) {
            // alert("success");
            var result = data.result[0];
            $('#result')[0].style.display="block";
        //   alert($("#table tr.selected td:first").html());
            if(result===1){
                $('#result').addClass('alert alert-success');
                $('#result')[0].textContent="Genuine Transaction";
            }
            else{
                $('#result').addClass('alert alert-danger');
                $('#result')[0].textContent="Fraudulent Transaction!";
            }
            document.getElementById("loading").innerHTML='';
        }
    })
})

$("#table tr").click(function () {
    $(this).siblings().removeClass('selected');
    $(this).addClass('selected')
    // var value = $(this).find('td').html();
    // var row = $(this).closest('td');
    json_obj=[];
    json="";
    for(var i=0;i<30;i++){
         json_obj.push($(this).find(`td:eq(${i})`).text());
    }
    json = Object.assign({}, json_obj);
    console.log(json);
    // alert(value);
});
