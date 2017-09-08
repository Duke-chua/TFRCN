function dataset = caltech_test(dataset, usage, use_flip)

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_caltech('./datasets/caltech', 'test', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, true), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_caltech('./datasets/caltech', 'test', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    case {'train30'}
        dataset.imdb_train    = {  imdb_from_caltech('./datasets/caltech', 'test30', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, true), dataset.imdb_train, 'UniformOutput', false);
    case {'test30'}
        dataset.imdb_test     = imdb_from_caltech('./datasets/caltech', 'test30', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    otherwise
        error('usage = ''train'' or ''test''');
end

end