


// loading clusters
axios.get('wild.json')
  .then(function (response) {

    create_data(response.data);
    console.log("clusters data loaded");
  })
  .catch(function (error) {
    console.log(error);
  });

function create_data(data) {
  var class_info = {}
  //console.log(data);
  for (var cid in data) {
    var class_data = data[cid];
    clas = class_data.class.toLowerCase();
    class_info[clas] = class_data;
  }
  
  
 window.vm = new Vue({
    el: '#data',
    data: {
      card_list: [],
      current_sorting_cluster: 0,
      displayAllDecks: true,
      selected: 'warrior',
        options: [
          { text: 'Druid', value: 'druid'},          
          { text: 'Hunter', value: 'hunter' },
          { text: 'Mage', value: 'mage'},
          { text: 'Rogue', value: 'rogue'},     
          { text: 'Paladin', value: 'paladin' },          
          { text: 'Priest', value: 'priest'},  
          { text: 'Shaman', value: 'shaman'},
          { text: 'Warrior', value: 'warrior' },
          { text: 'Warlock', value: 'warlock'}
        ],
        class_info: class_info,
        current_class: class_info['warrior'],
      },
    methods: {
      setClass:function() {
        var cls = class_selector.value;
        console.log("requested class:", cls);
        this.current_class = this.class_info[cls];
        this.current_sorting_cluster = 0;
      },
      setSortCluster:function(cluster_id) {
        this.current_sorting_cluster = cluster_id;
      }
    },
    computed: {
      //returns the current cards for the class
      cards: function() {
          //console.log(this.current_class);
          var info = this.current_class;
          var card_list_obj = {}
          for (var cid in info.card_list) {
            var card = info.card_list[cid];
            card_list_obj[card] = {
              name: card,
              data: new Array(info.num_clusters).fill({value: 0})
            };
          }
          //console.log(card_list);
          for (var cid in info.clusters) {
            //console.log(info.clusters[cid].num_observations);
              //cluster core card
              for (var card in info.clusters[cid].core_cards ){

                value = info.clusters[cid].core_cards[card];
                  if (this.displayAllDecks) {
                    ctype = info.clusters[cid].num_observations > 5000 ? "core": "core-faded";
                  } else {
                    ctype = info.clusters[cid].num_observations > 5000 ? "core": "invisible";
                  }
                  card_list_obj[card]['data'][cid] = {
                    type: ctype,
                    value: value
                  }
             
               }
              //cluster tech cards
              for (var card in info.clusters[cid].tech_cards ){
                  value = info.clusters[cid].tech_cards[card];
                  if (this.displayAllDecks) {
                    ctype = info.clusters[cid].num_observations > 5000 ? "tech": "tech-faded";
                  } else {
                    ctype = info.clusters[cid].num_observations > 5000 ? "tech": "invisible";
                  }

                  card_list_obj[card]['data'][cid] = {
                    type: ctype,
                    value: value
                  }
              }
          }
          this.card_list = Object.keys(card_list_obj).map(function(key){
                    return card_list_obj[key];
          });

          var current_sorting_cluster = this.current_sorting_cluster
          this.card_list.sort(function(a,b){
          //console.log(a);
          //console.log(cluster_id, b.data[cluster_id].value);
            return b.data[current_sorting_cluster].value - a.data[current_sorting_cluster].value;
          });
          return this.card_list;
      }
    }
  });
}