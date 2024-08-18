import json
from operator import itemgetter


with open("plook.json", "r") as input_file:
    plook_data = json.load(input_file)

    plook_data = plook_data[:10]


combined_result = []
for rec in plook_data:

    cid = rec['claim_id']
    for res in rec['result']:
        new_rec = {
                'claim_id': cid,
                'section:': res['section'],
                'distance': res['distance']
                }
        combined_result.append(new_rec)

combined_result = sorted(combined_result, key=itemgetter('distance'), reverse=False)

with open("plook_results.json", "w") as outfile:
    json.dump(combined_result, outfile, indent=6)
