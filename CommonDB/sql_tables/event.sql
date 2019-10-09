create table event
(
    eid      varchar(100) not null,
    dataset  varchar(100) not null,
    etype    varchar(100) null,
    start_dt datetime     null,
    end_dt   datetime     null,
    primary key (eid, dataset)
);
