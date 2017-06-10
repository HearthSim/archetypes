import json

data = json.loads(open("wild.json").read())
print "class\tcluster_id\tseen\tnum_deck\tprevalence\twin median\twin stddev\tcore_cards\ttech_cards"
for pclass, clusters in data.items():
    for cid, cinfo in clusters.items():
        core = ", ".join(cinfo['core_cards_name'])
        tech = ", ".join(cinfo['tech_cards_name'])
        print "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (pclass, cid, cinfo['observations'], cinfo['num_decks'], cinfo['prevalence'], round(cinfo['win_rate']['mean'], 2), round(cinfo['win_rate']['stddev'], 2), core, tech) 