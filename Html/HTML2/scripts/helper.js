/*
	Helper functions
*/


/*
	Ensuring data type is set up for CORS
*/
function g_ajaxer(url_str, params, ok_cb, fail_cb){
	$.ajax({
		url: url_str,
		type: "POST",
		data: JSON.stringify(params),
		crossDomain: true,
		contentType: "application/json",
		dataType: "json",
		success: ok_cb,
		error: fail_cb,
		timeout: 6000
	});
}
/**
 * g_loadTheCitiesIntoDropDown(
 * 
 * Just populates a list of cities
 *
 * @param JqueryObject $parent_drop_down_e
 * @param Function parent_cb
 * 
 * return via parent_cb //always true || fail hard
*/
function g_loadTheCitiesIntoDropDown($parent_drop_down_el, parent_cb){
	$.get("cities.md", function(city_str){
		var
			html_str = '',
			city_arr = [],
			model_arr = [],
			city_temp_arr = city_str.split("\n");
			model_temp_arr = city_str.split("\n");

		city_arr = city_temp_arr.map(function(item){
		  return item.split(",")[0];
		}
		);
		model_arr = model_temp_arr.map(function(item){
		  return item.split(",")[1];
		}
		
		);
		//html_str += '<option value="">' + 'Select Model' + '</option>';
		for(var i_int = 0; i_int < city_arr.length; i_int += 1){
			html_str += '<option value="' + model_arr[i_int] + '">' + city_arr[i_int] + '</option>';
		}
		
		$parent_drop_down_el.html(html_str);
		
		const choices = new Choices('[data-trigger]',
		{
			searchEnabled: false,
			itemSelectText: '',
		});

		
		parent_cb(true);//done
		//and if this fails, fail hard here instead
	});
}
/**
 * g_loadTheCitiesIntoDropDown(
 * 
 * Takes a city and temperature and replies with text
 * It decides if it is too hot or too cold
 * Range of temps is currently set between
 * 20 and 79
 *
 * @param String city_str
 * @param Number temp_int
 * 
 * @return String  //The temperature is 20 degree, 
 * 		//I think that is too cold for cats"
 */
function g_customizeResponse(city_str, input_str,  probability_int, country_region_str){
	var 
		message_str = "The name <font color='red'>" + input_str + "</font> has the probability of <font color='red'>" + probability_int.toString() + "%</font> originating from the following country/region: <font color='red'>"+country_region_str+"</font>" 
		// message_str += "<br /><br />";

	//if(temp_int > 72){
	//	message_str += " I think this is too hot for cats.";
	//}else if(temp_int <= 72 && temp_int > 50){
	//	message_str += " I think this is probably just right for your cat.";
	//}else if(temp_int <= 50 && temp_int >= 30){
	//	message_str += " I think this maybe a bit cold for your cat.";
	//}else{
	//	message_str += " I think this is far too cold for cats.";
	//}
	return message_str;
}




