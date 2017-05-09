function dataset = kaist_visible_trainval(dataset, usage)

switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_kaist_visible('./datasets/kaist_visible', 'train', false) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, false), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_kaist_visible('./datasets/kaist_visible', 'train', false) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    otherwise
        error('usage = ''train'' or ''test''');
end

end