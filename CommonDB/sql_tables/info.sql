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
