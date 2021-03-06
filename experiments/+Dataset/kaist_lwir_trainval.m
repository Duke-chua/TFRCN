function dataset = kaist_lwir_trainval(dataset, usage, use_flipped)

switch usage(1:4)
    case {'trai'}
        dataset.imdb_train    = {  imdb_from_kaist_lwir('./datasets/kaist', usage, use_flipped) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x, false), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_kaist_lwir('./datasets/kaist', 'train20', use_flipped) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, false);
    otherwise
        error('usage = ''train'' or ''test''');
end

end