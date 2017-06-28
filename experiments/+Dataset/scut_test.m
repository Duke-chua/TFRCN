function dataset = scut_test(dataset, usage)

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_scut('./datasets/scut', 'test', true) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, true), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_scut('./datasets/scut', 'test', false) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    case {'testall'}
        dataset.imdb_test     = imdb_from_scut('./datasets/scut', 'scut', false) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    case {'scut02'}
        dataset.imdb_test     = imdb_from_scut('./datasets/scut', 'scut02', false) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    otherwise
        error('usage = ''train'' or ''test''');
end

end