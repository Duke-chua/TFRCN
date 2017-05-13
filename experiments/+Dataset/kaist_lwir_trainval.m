function dataset = kaist_lwir_trainval(dataset, usage, flip)

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_kaist_lwir('./datasets/kaist_lwir', 'train', flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, flip), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_kaist_lwir('./datasets/kaist_lwir', 'train', flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, flip);
    otherwise
        error('usage = ''train'' or ''test''');
end

end