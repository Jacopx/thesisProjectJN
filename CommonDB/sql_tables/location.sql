create table location
(
    id        varchar(100) not null,
    dataset   varchar(100) not null,
    latitude  float        null,
    longitude float        null,
    primary key (id, dataset),
    constraint location_ibfk_1
        foreign key (id, dataset) references object (id, dataset)
);
