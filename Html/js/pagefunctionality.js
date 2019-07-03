  // Create parsing link
		
	  /**
	  * Genterate link of favorite list array to clipboard
	  * 
	  * @parameters array of favorite names
	  * @return nothing
	  * 
	  */
		function copyFavoriteListLinkToClipboard(array){
			var arrStr = encodeURIComponent(JSON.stringify(array));
			var dummy = document.createElement("textarea");
			// to avoid breaking orgain page when copying more words
			// cant copy when adding below this code
			// dummy.style.display = 'none'
			document.body.appendChild(dummy);
			//Be careful if you use texarea. setAttribute('value', value), which works with "input" does not work with "textarea". – Eduard
			dummy.value = "https://www.babyname.ai/?array=" + arrStr;
			dummy.select();
			document.execCommand("copy");
			document.body.removeChild(dummy);
		    /* Alert the copied text */
			alert("Copied the favorite list location to clipboard: " + "https://www.babyname.ai/?array=" + arrStr);
		}
	  /**
	  * Generate facebook single name link
	  * 
	  * @parameters string containing
	  * @return facebook link
	  * 
	  */
	  function getFacebookLinkSingleName(stringName){
		var quote = "What+do+you+think+of+the+baby+name+" + stringName + "?";
		var facebookSingleNameLink = '<a id="clickable" style="color: inherit;" href="https://www.facebook.com/sharer/sharer.php?u=https://www.babyname.ai/?array=%5B%22'+stringName+'%22%5D&quote='+ quote + '">';
		return facebookSingleNameLink;
	  }
	  
	  
	  /**
	  * Generate facebook single name link
	  * 
	  * @parameters string containing
	  * @return facebook link
	  * 
	  */
	  function getTwitterLinkSingleName(stringName){
		var quote = "What+do+you+think+of+the+baby+name+" + stringName+"?";
		var twitterSingleNameLink = '<a id="clickable" style="color: inherit;" href="https://twitter.com/intent/tweet?text='+ quote + '&url=' + encodeURI('https://www.babyname.ai/?array=%5B%22'+encodeURI(stringName)+'%22%5D') + '">';
		//var twitterSingleNameLink = '<a id="clickable" class="twitter-share-button" style="color: inherit;" href="https://twitter.com/intent/tweet" data-size="large" data-text="custom share text" data-url="https://dev.twitter.com/web/tweet-button" data-hashtags="example,demo" data-via="twitterdev" data-related="twitterapi,twitter">';
		return twitterSingleNameLink;
	  }
	  
	  /**
	  * populateResults
	  * 
	  * @parameters resultsArray
	  * @return nothing
	  * 
	  */	
	  
	  function populateResults(resultsArray){
	    var resultlist = document.getElementById("resultlist");
		resultlist.innerHTML="";
		for (var i =0; i < resultsArray.length;i++){
				resultlist.innerHTML += '<li id="result" class="resultli">'+resultsArray[i]+' <span class="hearticon"> <i class="far fa-heart"></i> </span> <span class="shareicon">'+ getFacebookLinkSingleName(resultsArray[i]) +'<i class="fab fa-facebook-f"></i></a></span><span class="shareicon">'+ getTwitterLinkSingleName(resultsArray[i]) +'<i class="fab fa-twitter"></i></a></span>	</li>';
				
		}
	  
	  }
		
	  /**
	  * checkForParsing
	  * 
	  * @parameters none
	  * @return nothing
	  * 
	  */	
	  
	  function checkforParsing(){
		// Parse shared names and populate names
			var urlParams = new URLSearchParams(window.location.search);
			var resultlist = document.getElementById("resultlist");
			const myParam = urlParams.get('array');
			if (myParam) {
				parameterArray=JSON.parse(myParam);
				resultlist.innerHTML="";
				populateResults(parameterArray);
			}else{
				exampleArray=["Emma","Olivia", "Ava", "Isabella", "Sophia", "Charlotte", "Mia", "Amelia"];
				resultlist.innerHTML="";
				populateResults(exampleArray);
			}
		
		}

				
	  /**
	  * populateFavorites
	  * 
	  * @parameters populates the favorite list
	  * @return nothing
	  * 
	  */
	  function populateFavorites(favoriteArray) {
		/*Set array for testing*/
		/*favoriteArray=["Christer", "Øyvind", "Reinhart"];*/
		var favoirteList = document.getElementById("favoriteList");
		favoirteList.innerHTML ="";
		nofavoritestext = document.getElementById("nofavoritestext");
		opyfavoritelinklocation = document.getElementById("copyfavoritelinklocation");
		
		/*Check if favorites are empty*/
		if (favoriteArray.length==0){
			nofavoritestext.style.display = "block";
			copyfavoritelinklocation.style.display = "none";			
		}else{
			nofavoritestext.style.display = "none";
			copyfavoritelinklocation.style.display = "block";
		}
		
		/*Favorite Array*/
		for (var i =0; i < favoriteArray.length;i++){
			favoirteList.innerHTML += '<li class="favoriteListItem">'+favoriteArray[i].trim()+' <span class="hearticon_favorites" name="'+favoriteArray[i].trim()+'"> <i class="fas fa-heart"></i> </span>  <span class="shareicon">'+ getFacebookLinkSingleName(favoriteArray[i].trim()) +'<i class="fab fa-facebook-f"></i></a></span><span class="shareicon">'+ getTwitterLinkSingleName(favoriteArray[i].trim()) +'<i class="fab fa-twitter"></i></a></span></li>';
			
			var content = document.getElementById("favorite");
			if (content.style.maxHeight){
				content.style.maxHeight = content.scrollHeight + "px";
			} else {
			} 
		}
		
	   }
	   
	   /**
	  * checkForStoredFavoritess
	  * 
	  * @parameters none
	  * @return nothing
	  * 
	  */
	   function checkForStoredFavorites(){
			var favoriteArray = [];
			var savedFavoriteArray = localStorage.getItem('favoriteArray');

			// If there are any saved items, update our list
			if (savedFavoriteArray) {
				favoriteArray = JSON.parse(savedFavoriteArray);
				populateFavorites(favoriteArray);
				//favoriteList.innerHTML = saved;
			};
	   }
	   
	  /**
	  * contains
	  * 
	  * @parameters a (array), obj (value) 
	  * @return true or false whether what the values are
	  * 
	  */	   
	  function contains(a, obj) {
		for (var i = 0; i < a.length; i++) {
			if (a[i] === obj) {
				return true;
			}
		}
		return false;
	  } 
	   
	   
	  /**
	  * addToFavorite
	  * 
	  * @parameters favoriteArray and stringToAdd
	  * @return nothing
	  * 
	  */
	   
		function addToFavorite(favoriteArray, stringToAdd){
			/*Check if value exist from before*/
			if (contains(favoriteArray,stringToAdd.trim())){
			}else{
				favoriteArray.push(stringToAdd.trim());
				localStorage.setItem('favoriteArray', JSON.stringify(favoriteArray));
			}
		}
		
		/**
	  * removeFromFavorite
	  * 
	  * @parameters favoriteArray and stringToAdd
	  * @return nothing
	  * 
	  */	   
		function removeFromFavorite(favoriteArray, stringToRemove){
			/*Check if value exist from before*/
			if (contains(favoriteArray,stringToRemove)){
				for (var i = 0; i < favoriteArray.length; i++) {
					if (favoriteArray[i] === stringToRemove.trim()) {
						favoriteArray.splice(i, 1); 
					}
				}
				localStorage.setItem('favoriteArray', JSON.stringify(favoriteArray));
			}else{
			}
		}
	
	  /**
	  * getFavoriteArray
	  * 
	  * @parameters None
	  * @return favoriteArray
	  * 
	  */
	  function getFavoriteArray(){
			var favoriteArray = [];
			var savedFavoriteArray = localStorage.getItem('favoriteArray');

			// If there are any saved items, update our list
			if (savedFavoriteArray) {
				favoriteArray = JSON.parse(savedFavoriteArray);
				return favoriteArray;
			}else{
				return [];
			}
				//favoriteList.innerHTML = saved;
	   }
	
	  /**
	  * addListenerToResults
	  * 
	  * @parameters none
	  * @return nothing
	  * 
	  */
	  
	 function addListenerToFavorites(){
		var item = document.getElementsByClassName("hearticon_favorites");
		for (var i = 0; i < item.length; i++) {
			item[i].addEventListener('click', function() {
				removeFromFavorite(getFavoriteArray() ,this.getAttribute("name"));
				populateFavorites(getFavoriteArray());
				addListenerToFavorites();
			});
		}
	}
		
	  
	 function addListenerToResults(){
		var item = document.getElementsByClassName("resultli");
		for (var i = 0; i < item.length; i++) {
			item[i].addEventListener('click', function() {
			  const icon = this.querySelector('i');
			  if (icon.classList.contains('far','fa-heart')) {
				icon.classList.remove('far','fa-heart');
				icon.classList.add('fas', 'fa-heart');
				addToFavorite(getFavoriteArray() ,this.textContent.trim());
				populateFavorites(getFavoriteArray());
				addListenerToFavorites();
			  } else {
				icon.classList.remove('fas','fa-heart');
				icon.classList.add('far', 'fa-heart');
				removeFromFavorite(getFavoriteArray() ,this.textContent.trim());
				populateFavorites(getFavoriteArray());
				addListenerToFavorites();
			  }
			});
		};
	}
	
	
	function addListenerToCopyToClipboard(){
		var item = document.getElementsByClassName("copyfavoritelist");
		for (var i = 0; i < item.length; i++) {
			item[i].addEventListener('click', function() {
				copyFavoriteListLinkToClipboard(getFavoriteArray());
			  
			  });
		}
	 
	}
		
	
		
	   checkforParsing();
	   checkForStoredFavorites();
	   addListenerToResults();
	   addListenerToFavorites();
	   addListenerToCopyToClipboard();
	
	function configureResultsDisplay(){
		
		var shown_at_start=5;
		var items =  $('#resultlist li').length;
		
		if (items  <= shown_at_start){
			$('#loadMore').hide();
			$('#showLess').hide();
		}
		else{
			$('#loadMore').show();
		}
		
		$('#resultlist li:lt(5)').show();
		$('#showLess').hide();
				
		
		$('#loadMore').click(function () {
			$('#showLess').show();
			shown = $('#resultlist li:visible').length+shown_at_start;
			if(shown<$('#resultlist li').length){
				$('#resultlist li:lt('+shown+')').show();
			}else {
				$('#resultlist li:lt('+$('#resultlist li').length+')').show();
				$('#loadMore').hide();
			}
		});
		$('#showLess').click(function () {
			$('#resultlist li').not(':lt(5)').hide();
			$('#loadMore').show();
			$('#showLess').hide();
		});
	}
	
	
	$(document).ready(function () {
		// Load the first 5 list items from another HTML file
		//$('#myList').load('externalList.html li:lt(5)');
		configureResultsDisplay();
		
	});

	$("#resultlist a").click(function(e) {
		e.stopPropagation();
	});