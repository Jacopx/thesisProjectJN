create table object
(
    id      varchar(100) not null,
    dataset varchar(100) not null,
    type    varchar(255) null,
    primary key (id, dataset)
);
