import csv

with open('submission.csv', 'r') as f:
    fp = csv.reader(f)
    contoursdata = {}
    for row in fp:
        vals = row
        scan_id = vals[0]
        if scan_id not in contoursdata:
            contoursdata[scan_id] = {}
            contoursdata[scan_id][int(vals[1])] = vals[2:]
        elif vals[1] not in contoursdata[scan_id]:
            contoursdata[scan_id][int(vals[1])] = vals[2:]

    with open('submission_processed.csv', 'w+') as outcsv:
        for key1 in contoursdata.keys():
            keyView = contoursdata[key1].keys()
            for key2 in keyView:
                if key2 + 1 in keyView or key2 - 1 in keyView or key2 + 2 in keyView or key2 - 2 in keyView:
                    s = key1 + ',' + str(key2)
                    vals = contoursdata[key1][key2]
                    for i in range(0, len(vals), 2):
                        s += ',' + vals[i] + ',' + vals[i + 1]
                    outcsv.write(s + '\n')
    outcsv.close()
