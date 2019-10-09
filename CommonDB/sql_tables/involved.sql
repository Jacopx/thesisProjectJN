create table involved
(
    eid     varchar(100) not null,
    dataset varchar(100) not null,
    id      varchar(100) not null,
    primary key (eid, dataset, id),
    constraint involved_ibfk_1
        foreign key (eid, dataset) references event (eid, dataset),
    constraint involved_ibfk_2
        foreign key (id, dataset) references object (id, dataset)
);

create index id
    on involved (id, dataset);
