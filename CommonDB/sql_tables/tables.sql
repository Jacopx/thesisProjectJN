create table event
(
    eid      varchar(100) not null,
    dataset  varchar(100) not null,
    etype    varchar(100) null,
    start_dt datetime     null,
    end_dt   datetime     null,
    primary key (eid, dataset)
);

create table object
(
    id      varchar(100) not null,
    dataset varchar(100) not null,
    type    varchar(255) null,
    primary key (id, dataset)
);

create table involved
(
    eid     varchar(100) not null,
    dataset varchar(100) not null,
    type    varchar(100) not null,
    id      varchar(100) not null,
    primary key (eid, dataset, type, id),
    constraint involved_ibfk_1
        foreign key (eid, dataset) references event (eid, dataset),
    constraint involved_ibfk_2
        foreign key (id, dataset) references object (id, dataset)
);

create index id
    on involved (id, dataset);

create table info
(
    id      varchar(100)  not null,
    dataset varchar(100)  not null,
    type    varchar(255)  null,
    descr   varchar(1000) null,
    primary key (id, dataset),
    constraint info_ibfk_1
        foreign key (id, dataset) references object (id, dataset)
);

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
