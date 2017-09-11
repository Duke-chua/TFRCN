function dataset = scut_test(dataset, usage, use_flip)

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_scut('./datasets/scut', 'test', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, use_flip), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_scut('./datasets/scut', 'test', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, use_flip);
    otherwise
        error('usage = ''train'' or ''test''');
end
end