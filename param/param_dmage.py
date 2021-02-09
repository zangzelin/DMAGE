import argparse



def GetParamCora():

    parser = argparse.ArgumentParser(description='zelin zang author')
    parser.add_argument('--name', type=str, default='cora_', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='DMAGE', )
    parser.add_argument('--data_name', type=str, default='cora', )
    parser.add_argument('--data_trai_n', type=int, default=2708, )

    # model param
    parser.add_argument('--perplexity', type=int, default=50, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--alpha', type=float, default=1.0, )
    parser.add_argument('--NetworkStructure', type=list, default=[1000, 500, 250, 200], )
    parser.add_argument('--dropedgerate', type=float, default=0.01, )
    parser.add_argument('--model_type', type=str, default='DMAGE', )

    # train param
    parser.add_argument('--batch_size', type=int, default=2708, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    args['vtrace'] = [args['vs'],args['ve']]

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args


def GetParamCiteSeer():

    parser = argparse.ArgumentParser(description='zelin zang author')
    parser.add_argument('--name', type=str, default='citeseer_', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='DMAGE', )
    parser.add_argument('--data_name', type=str, default='citeseer', )
    parser.add_argument('--data_trai_n', type=int, default=3327, )

    # model param
    parser.add_argument('--perplexity', type=int, default=80, )
    parser.add_argument('--vs', type=float, default=0.0034, )
    parser.add_argument('--ve', type=float, default=0.0034, )
    parser.add_argument('--alpha', type=float, default=0.5, )
    parser.add_argument('--NetworkStructure', type=list, default=[1000, 500, 250, 200], )
    parser.add_argument('--dropedgerate', type=float, default=0.01, )
    parser.add_argument('--model_type', type=str, default='DMAGE', )

    # train param
    parser.add_argument('--batch_size', type=int, default=3327, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=50 )
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    args['vtrace'] = [args['vs'],args['ve']]

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args

def GetParamPubMed():

    parser = argparse.ArgumentParser(description='zelin zang author')
    parser.add_argument('--name', type=str, default='pubmed_', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='DMAGE', )
    parser.add_argument('--data_name', type=str, default='pubmed', )
    parser.add_argument('--data_trai_n', type=int, default=20000, )

    # model param
    parser.add_argument('--perplexity', type=int, default=50, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--alpha', type=float, default=1.0, )
    parser.add_argument('--NetworkStructure', type=list, default=[1000, 500, 250, 200], )
    parser.add_argument('--dropedgerate', type=float, default=0.01, )
    parser.add_argument('--model_type', type=str, default='DMAGE', )

    # train param
    parser.add_argument('--batch_size', type=int, default=5000, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    args['vtrace'] = [args['vs'],args['ve']]

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args