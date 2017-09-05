function dataset = scut_trainval(dataset, usage, use_flip)

devkit = scut_devkit();

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_scut(devkit, 'train04', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, false), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_scut(devkit, 'train04', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);

    otherwise
        error('usage = ''train'' or ''test''');
end
end