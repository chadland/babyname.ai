/*
	Requires G_API_GATEWAY_URL_STR in the window object, 
	see readme and exercise guide
*/
var 
	$button = $("button"),
	$select = $("select"),
	$output = $("output"),
	$input = $("input");

/**
 * _askTheFakeBot
 *
 * This is a FAKE CLIENT chatbot
 * It does not hit a backend
 * "regardless of which city you pass",
 * it thinks it is 20 degrees
 * and is too cold for a cat.
 * 
 * @param String city_str // austin
 * @return To Callback Via Custom Reponse Helper
 * 		//response_str
 */
function _askTheFakeBot(city_str, input_str, cb){
	var 
		hard_coded_temp_int = 20.88,
		response_str = "";
	cb(g_customizeResponse(city_str, input_str, hard_coded_temp_int));
}

/**
 * _askTheMockAPIBot
 *
 * This is a API chatbot
 * It DOES hit an API
 * If you have just wired up the api gateway mock
 * it will thinks everywhere is 69 degrees
 * and it is probably just right for your cat.
 * 
 * 
 * If you have just wired up the api gateway LAMBDA mock
 * it will thinks everywhere is 76 degrees
 * and it is probably too hot for your cat.
 * 
 * If you have just wired up the api gateway LAMBDA to
 * DynamoDB
 * then it will return a different temp per city.
 * and the message may change per city.
 * @param String city_str // austin
 * @return To Callback Via Custom Reponse Helper
 * 		//response_str
 */
function _askTheAPIBot(city_str, input_str, cb){
	console.log("We are hitting the API: " + G_API_GATEWAY_URL_STR);
	console.log("Model String: " + city_str);
	var 
		params = {
			"first_name":input_str,
			"model_name":city_str
		};
	console.log("Params: " + params.first_name);
	console.log("Params: " + params.city_str);
	g_ajaxer(G_API_GATEWAY_URL_STR, params, function(response){
		handleGatewayResponse(response, city_str, input_str, cb);
	}, function(error){
		handleGatewayError(error, cb);
	});
}

function handleGatewayResponse(response, city_str,input_str, cb){
	var 
		probability_int = response.probability_int;
	var 
		country_region_str = response.country_region_str;

	console.log("Probability: " + probability_int.toString());
	console.log("Country: " + country_region_str.toString());
	cb(g_customizeResponse(city_str, input_str, probability_int, country_region_str));
}

function handleGatewayError(error, cb){
	cb("This failed:" + error.statusText);
}
/**
 * _askTheBot
 *
 * Proxy to the right bot
 *  
 * @param String city_str // austin
 * @param Function //parent_cb
 */
function _pickABot(city_str,input_str, cb){
	if(city_str === ""){
		response_str = "Please pick a model type, thanks";
		return cb(response_str);
	}
	if(G_API_GATEWAY_URL_STR === null){
		_askTheFakeBot(city_str, input_str, cb);
	}else{
		_askTheAPIBot(city_str,input_str, cb);
	}
}
/**
 * whatShouldMyPetDo
 * 
 * @param Submit Event from form
 * @return undefined //UI change on output
 * 
 */
function whatShouldMyPetDo(se){
	var 
		city_str = "";
	var 
		search_str="";
	se.preventDefault();
	if($button.prop("disabled") === true){
		return;
	}
	$output.attr("data-showing", "not_showing");
	$button.prop("disabled", "true");
	city_str = $select.val();//they are already uppercase
	input_str = $input.val();
	_pickABot(city_str,input_str, function(response_str){
		$output.html(response_str);
		$output.attr("data-showing", "showing");
		$button.prop("disabled", false);
	});
}

//only start the app once we have all the cities
g_loadTheCitiesIntoDropDown($select, function(){
	$button.prop("disabled", false);
});


//handlers
$(document).on("submit", whatShouldMyPetDo);
