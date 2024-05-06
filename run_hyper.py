import argparse
import os

from recbole.trainer import HyperTuning
from recbole_gnn.quick_start import objective_function as recbole_objective_function
from main import objective_function as my_objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='', help='fixed config files')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='output file')
    parser.add_argument('--hyper_early_stop', type=int, default=10, help='hyper early stop')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set

    config_file_list = [
        'properties/overall.yaml',
    ]
    if args.config_files != '':
        config_file_list.extend(args.config_files.split(","))
    if args.dataset in ['yelp', 'amazon-books', 'amazon-kindle-store', 'QB-video']:
        config_file_list.append(f'properties/{args.dataset}.yaml')

    # the difference of objective_function is trainer
    hp = HyperTuning(recbole_objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list, early_stop=args.hyper_early_stop)
    hp.run()
    output_results = os.path.join("results", args.output_file)
    hp.export_result(output_file=output_results)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
